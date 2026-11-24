"""DPO (Direct Preference Optimization) — 교정 기록의 (교사 정답, 학생 오답) 쌍으로 "그거 말고 이거"를 가르친다.

    train_dpo(base="distill_out/r3/merged", pairs="pairs.jsonl", out_dir="distill_out/r4", beta=0.1)

- 정책 = base(SFT 를 합친 모델) + 새 LoRA (학습), 기준 = 같은 base 에서 어댑터를 끈 것 (메모리 한 벌).
- 손실 = -log σ(β · [(log π(c) − log π_ref(c)) − (log π(r) − log π_ref(r))]), log π 는 assistant 토큰만 합.
- 쌍 데이터 형식 btqa-play-pref/1 (btqa_play.dataset.build_pairs).
- 출력: <out_dir>/adapter (LoRA), dpo_log.json. 실행 모델로 쓰려면 export.merge(base=<SFT merged>, adapter=<out_dir>/adapter).
"""
from __future__ import annotations

import json
import random
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple, Union

from btqa_distill.config import QWEN_TARGETS, load_causal_lm
from btqa_distill.train import _ids


def _encode_pair(tok: Any, prompt: List[Dict[str, str]], answer: str) -> Tuple[List[int], int]:
    full = _ids(tok.apply_chat_template(prompt + [{"role": "assistant", "content": answer}], tokenize=True))
    p = _ids(tok.apply_chat_template(prompt, tokenize=True, add_generation_prompt=True))
    if full[:len(p)] != p:
        raise ValueError("chat template 이 프롬프트를 접두사로 만들지 않는다")
    return full, len(p)


def _seq_logp(model: Any, ids: List[int], n_prompt: int, torch: Any, device: str, dtype: Any) -> Any:
    x = torch.tensor([ids], device=device)
    with torch.autocast(device_type="cuda" if device.startswith("cuda") else "cpu", dtype=dtype, enabled=device.startswith("cuda")):
        logits = model(input_ids=x).logits[:, :-1, :].float()
    tgt = x[:, 1:]
    lp = torch.log_softmax(logits, dim=-1).gather(-1, tgt.unsqueeze(-1)).squeeze(-1)
    return lp[:, n_prompt - 1:].sum()


def train_dpo(base: str, pairs: Union[str, Path], out_dir: Union[str, Path], beta: float = 0.1, lr: float = 2e-5,
              epochs: int = 2, grad_accum: int = 8, lora_r: int = 16, lora_alpha: int = 32, max_len: int = 4096,
              dtype_name: str = "bfloat16", seed: int = 0) -> Path:
    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import AutoTokenizer

    random.seed(seed)
    torch.manual_seed(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = getattr(torch, dtype_name)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    tok = AutoTokenizer.from_pretrained(base)
    rows = [json.loads(x) for x in Path(pairs).read_text(encoding="utf-8").splitlines() if x.strip()]
    data = []
    for r in rows:
        c, nc = _encode_pair(tok, r["prompt"], r["chosen"])
        j, nj = _encode_pair(tok, r["prompt"], r["rejected"])
        if max(len(c), len(j)) <= max_len:
            data.append((c, nc, j, nj))
    if not data:
        raise ValueError("DPO 쌍이 없다")

    model = load_causal_lm(base, dtype)
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    model = get_peft_model(model, LoraConfig(r=lora_r, lora_alpha=lora_alpha, lora_dropout=0.0,
                                             target_modules=list(QWEN_TARGETS), task_type="CAUSAL_LM"))
    for p in model.parameters():
        if p.requires_grad:
            p.data = p.data.float()
    model.to(device)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=lr)

    # 기준 log π_ref 는 미리 한 번 (어댑터 끔)
    ref = []
    model.eval()
    with torch.no_grad(), model.disable_adapter():
        for c, nc, j, nj in data:
            ref.append((_seq_logp(model, c, nc, torch, device, dtype).item(), _seq_logp(model, j, nj, torch, device, dtype).item()))
    model.train()
    log: Dict[str, Any] = {"base": base, "pairs": len(data), "beta": beta, "lr": lr, "epochs": epochs, "steps": []}
    t0, step = time.time(), 0
    idx = list(range(len(data)))
    for ep in range(epochs):
        random.shuffle(idx)
        opt.zero_grad()
        acc, losses = 0, []
        for k, i in enumerate(idx):
            c, nc, j, nj = data[i]
            rc, rj = ref[i]
            lc = _seq_logp(model, c, nc, torch, device, dtype)
            lj = _seq_logp(model, j, nj, torch, device, dtype)
            margin = beta * ((lc - rc) - (lj - rj))
            loss = -torch.nn.functional.logsigmoid(margin) / grad_accum
            loss.backward()
            losses.append(loss.item() * grad_accum)
            acc += int(margin.item() > 0)
            if (k + 1) % grad_accum == 0 or k == len(idx) - 1:
                torch.nn.utils.clip_grad_norm_(params, 1.0)
                opt.step()
                opt.zero_grad()
                step += 1
        rec = {"epoch": ep, "loss": round(sum(losses) / len(losses), 4), "pref_acc": round(acc / len(idx), 3),
               "seconds": round(time.time() - t0, 1)}
        log["steps"].append(rec)
        print(json.dumps(rec), flush=True)
    model.save_pretrained(str(out / "adapter"))
    tok.save_pretrained(str(out / "adapter"))
    (out / "dpo_log.json").write_text(json.dumps(log, ensure_ascii=False, indent=2), encoding="utf-8")
    return out / "adapter"


__all__ = ["train_dpo"]


if __name__ == "__main__":   # 간단 실행: python -m btqa_distill.dpo <base> <pairs.jsonl> <out_dir>
    import sys
    print(train_dpo(sys.argv[1], sys.argv[2], sys.argv[3]))
