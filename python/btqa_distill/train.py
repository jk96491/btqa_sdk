"""LoRA 학습 (GPU 권장) — btqa_play.dataset 의 train.jsonl / val.jsonl → PEFT 어댑터.

- loss 는 **assistant 출력(JSON 행동)에만** 건다: 토큰화 = chat template(전체) 에서 chat template(system+user, 생성 프롬프트)
  길이만큼을 -100 으로. assistant 끝 토큰(<|im_end|> 등)까지 배운다 → 생성이 멈춘다.
- max_len 보다 긴 샘플은 자르지 않고 뺀다 (system 이 잘리면 실행 때와 입력이 달라진다). 뺀 수는 로그에.
- 기반 가중치 = cfg.dtype(bf16 권장), LoRA 파라미터 = float32 + autocast. QLoRA(4bit)는 쓰지 않는다 (Windows bitsandbytes 회피,
  1.5B~3B 는 12GB 에 bf16 로 들어간다).
- 출력: <out_dir>/adapter/ (PEFT), tokenizer, train_log.json (손실·설정·뺀 샘플). **out_dir 은 git 제외 경로**로.

필요: torch(CUDA), transformers, peft — btqa_sdk/python/btqa_distill/requirements.txt (별도 venv, Python ≥ 3.10).
"""
from __future__ import annotations

import json
import math
import random
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

from btqa_distill.config import TrainConfig, load_causal_lm
from btqa_play.dataset import read_samples, split_prompt


def _ids(x: Any) -> List[int]:
    """apply_chat_template 결과 → 토큰 목록 (transformers 5 는 BatchEncoding/dict 를 돌려준다)."""
    if hasattr(x, "keys") and "input_ids" in x.keys():
        x = x["input_ids"]
    x = list(x)
    if x and isinstance(x[0], (list, tuple)):   # 배치 1개
        x = list(x[0])
    return [int(t) for t in x]


def encode(tok: Any, sample: Dict[str, Any]) -> Tuple[List[int], List[int]]:
    """(input_ids, labels) — labels 는 프롬프트 부분 -100."""
    prompt_msgs, target = split_prompt(sample)
    full = _ids(tok.apply_chat_template(prompt_msgs + [{"role": "assistant", "content": target}], tokenize=True))
    prompt = _ids(tok.apply_chat_template(prompt_msgs, tokenize=True, add_generation_prompt=True))
    if list(full[:len(prompt)]) != list(prompt):
        raise ValueError("chat template 이 프롬프트를 접두사로 만들지 않는다 — 이 모델 템플릿은 지원 밖")
    full = list(full)
    return full, [-100] * len(prompt) + full[len(prompt):]


def _batches(items: Sequence[Tuple[List[int], List[int]]], size: int, rng: random.Random, shuffle: bool):
    idx = list(range(len(items)))
    if shuffle:
        rng.shuffle(idx)
    for i in range(0, len(idx), size):
        yield [items[j] for j in idx[i:i + size]]


def answer_loss(model: Any, ids: Any, att: Any, lab: Any, torch: Any, device: str, dtype: Any, use_amp: bool) -> Any:
    """답(assistant) 토큰의 교차 엔트로피. logits 는 답 구간만 만든다 (`logits_to_keep`).

    입력 수천 토큰 × 어휘 15만의 전체 logits(fp32 1.8GB+, 기울기 포함 수 GB)를 만들지 않는다 — 3060 12GB 에서
    전체 logits 는 GPU 메모리를 넘겨 시스템 RAM 으로 넘치고 학습이 수 배 느려졌다 (2026-11-25 r4).
    값은 `model(labels=lab).loss` 와 같다 (라벨 있는 토큰 평균)."""
    T = ids.shape[1]
    labeled = (lab != -100).any(0).nonzero()
    p0 = int(labeled[0]) if len(labeled) else T - 1          # 처음 라벨 위치 (배치 전체)
    k = T - p0 + 1                                           # 위치 p0-1 ~ T-1 의 logits (p0-1 이 p0 를 예측)
    with torch.autocast(device_type="cuda" if device.startswith("cuda") else "cpu", dtype=dtype, enabled=use_amp):
        logits = model(input_ids=ids, attention_mask=att, logits_to_keep=k).logits
    pred = logits[:, :-1, :].float()
    tgt = lab[:, T - k + 1:]
    return torch.nn.functional.cross_entropy(pred.reshape(-1, pred.shape[-1]), tgt.reshape(-1), ignore_index=-100)


def _collate(batch, pad_id: int, torch: Any, device: str):
    n = max(len(x) for x, _ in batch)
    ids = torch.full((len(batch), n), pad_id, dtype=torch.long)
    lab = torch.full((len(batch), n), -100, dtype=torch.long)
    att = torch.zeros((len(batch), n), dtype=torch.long)
    for k, (x, y) in enumerate(batch):
        ids[k, :len(x)] = torch.tensor(x)
        lab[k, :len(y)] = torch.tensor(y)
        att[k, :len(x)] = 1
    return ids.to(device), att.to(device), lab.to(device)


def train_lora(cfg: TrainConfig, train_path: Union[str, Path], val_path: Optional[Union[str, Path]] = None,
               log_every: int = 10) -> Path:
    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import AutoTokenizer

    cfg.validate()
    random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    device = cfg.device or ("cuda" if torch.cuda.is_available() else "cpu")
    dtype = getattr(torch, cfg.dtype)
    out = Path(cfg.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    tok = AutoTokenizer.from_pretrained(cfg.base_model)
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id

    def load(path):
        kept, dropped = [], 0
        for s in read_samples(path):
            x, y = encode(tok, s)
            if len(x) > cfg.max_len:
                dropped += 1
                continue
            kept.append((x, y))
        return kept, dropped

    train, train_dropped = load(train_path)
    val, val_dropped = load(val_path) if val_path and Path(val_path).exists() else ([], 0)
    if not train:
        raise ValueError("학습 샘플이 없다 (max_len=%d 보다 모두 길거나 파일이 비었다)" % cfg.max_len)

    model = load_causal_lm(cfg.base_model, dtype)
    if cfg.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()
    model = get_peft_model(model, LoraConfig(r=cfg.lora_r, lora_alpha=cfg.lora_alpha, lora_dropout=cfg.lora_dropout,
                                             target_modules=list(cfg.target_modules), task_type="CAUSAL_LM"))
    for p in model.parameters():
        if p.requires_grad:
            p.data = p.data.float()
    model.to(device)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=cfg.lr, weight_decay=cfg.weight_decay)
    steps_per_epoch = math.ceil(len(train) / cfg.batch_size / cfg.grad_accum)
    total = max(1, steps_per_epoch * cfg.epochs)
    warm = int(total * cfg.warmup_ratio)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: (s + 1) / max(1, warm) if s < warm else max(0.0, (total - s) / max(1, total - warm)))
    use_amp = device.startswith("cuda") and cfg.dtype != "float32"
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp and cfg.dtype == "float16")
    rng = random.Random(cfg.seed)

    def run_loss(batch):
        ids, att, lab = _collate(batch, pad_id, torch, device)
        return answer_loss(model, ids, att, lab, torch, device, dtype, use_amp)

    log: Dict[str, Any] = {"config": cfg.to_dict(), "train_samples": len(train), "val_samples": len(val),
                           "dropped_too_long": {"train": train_dropped, "val": val_dropped}, "steps": [], "epochs": []}
    t0, step = time.time(), 0
    model.train()
    for epoch in range(cfg.epochs):
        opt.zero_grad()
        for i, batch in enumerate(_batches(train, cfg.batch_size, rng, shuffle=True)):
            loss = run_loss(batch) / cfg.grad_accum
            scaler.scale(loss).backward()
            if (i + 1) % cfg.grad_accum == 0:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(params, 1.0)
                scaler.step(opt)
                scaler.update()
                opt.zero_grad()
                sched.step()
                step += 1
                if step % log_every == 0:
                    rec = {"step": step, "epoch": epoch, "loss": round(loss.item() * cfg.grad_accum, 4),
                           "lr": sched.get_last_lr()[0], "seconds": round(time.time() - t0, 1)}
                    log["steps"].append(rec)
                    print(json.dumps(rec), flush=True)
        val_loss = None
        if val:
            model.eval()
            with torch.no_grad():
                losses = [run_loss(b).item() for b in _batches(val, cfg.batch_size, rng, shuffle=False)]
            val_loss = round(sum(losses) / len(losses), 4)
            model.train()
        log["epochs"].append({"epoch": epoch, "val_loss": val_loss, "seconds": round(time.time() - t0, 1)})
        print(json.dumps(log["epochs"][-1]), flush=True)

    model.save_pretrained(str(out / "adapter"))
    tok.save_pretrained(str(out / "adapter"))
    log["seconds"] = round(time.time() - t0, 1)
    (out / "train_log.json").write_text(json.dumps(log, ensure_ascii=False, indent=2), encoding="utf-8")
    return out / "adapter"


__all__ = ["train_lora", "encode"]
