"""DPO (Direct Preference Optimization) — (좋은 글, 나쁜 글) 쌍으로 "그거 말고 이거"를 가르친다.

    train_dpo(base="distill_out/r8/merged", pairs="pairs_onpolicy.jsonl", out_dir="distill_out/r9", beta=0.1, nll_coef=0.2)

- 정책 = base(SFT 를 합친 모델) + 새 LoRA (학습), 기준 = 같은 base 에서 어댑터를 끈 것 (메모리 한 벌).
- 손실 = −log σ(β·[(log π(c) − log π_ref(c)) − (log π(r) − log π_ref(r))])  +  nll_coef · NLL_토큰평균(c)
  두 번째 항(RPO 방식)은 chosen 의 확률 자체를 붙잡는다. DPO 만 쓰면 c·r 확률이 **같이** 떨어지면서 차이만 벌어져
  탐욕 디코딩이 엉뚱한 글로 넘어갈 수 있다 (AI_QA r9: 선호 정확도 1.0 인데 플레이 1/5).
- 쌍 데이터 형식 btqa-play-pref/1. 기본 출처는 on-policy 쌍 (btqa_distill.onpolicy) — 교정 기록 쌍(btqa_play.dataset.build_pairs)은
  chosen 이 교사 글이라 학생 분포 밖이다.
- val_pairs 를 주면 epoch 마다 검증 쌍의 선호 정확도·마진. 선호 정확도는 **플레이 품질이 아니다** → 채택은 평가 스위트로
  (btqa_play.suite.compare). 그래서 epoch 마다 어댑터를 따로 저장한다 (adapter-ep<N>/, 마지막은 adapter/).
"""
from __future__ import annotations

import json
import random
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from btqa_distill.train import _ids


def _encode_pair(tok: Any, prompt: List[Dict[str, str]], answer: str) -> Tuple[List[int], int]:
    full = _ids(tok.apply_chat_template(prompt + [{"role": "assistant", "content": answer}], tokenize=True))
    p = _ids(tok.apply_chat_template(prompt, tokenize=True, add_generation_prompt=True))
    if full[:len(p)] != p:
        raise ValueError("chat template 이 프롬프트를 접두사로 만들지 않는다")
    return full, len(p)


def _load(tok: Any, path: Union[str, Path], max_len: int) -> List[Tuple[List[int], int, List[int], int]]:
    rows = [json.loads(x) for x in Path(path).read_text(encoding="utf-8").splitlines() if x.strip()]
    data = []
    for r in rows:
        c, nc = _encode_pair(tok, r["prompt"], r["chosen"])
        j, nj = _encode_pair(tok, r["prompt"], r["rejected"])
        if max(len(c), len(j)) <= max_len:
            data.append((c, nc, j, nj))
    return data


def train_dpo(base: str, pairs: Union[str, Path], out_dir: Union[str, Path], beta: float = 0.1, lr: float = 5e-6,
              epochs: int = 2, grad_accum: int = 8, lora_r: int = 16, lora_alpha: int = 32, max_len: int = 4096,
              dtype_name: str = "bfloat16", seed: int = 0, nll_coef: float = 0.2,
              val_pairs: Optional[Union[str, Path]] = None) -> Path:
    import torch
    from btqa_distill.policy import LoraPolicy

    random.seed(seed)
    torch.manual_seed(seed)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    pol = LoraPolicy.load(base, lora_r=lora_r, lora_alpha=lora_alpha, dtype_name=dtype_name, train=True)
    tok = pol.tok
    data = _load(tok, pairs, max_len)
    val = _load(tok, val_pairs, max_len) if val_pairs else []
    if not data:
        raise ValueError("DPO 쌍이 없다")

    def seq(ids: List[int], n: int, mean: bool = False) -> Any:
        return pol.logp(ids[:n], ids[n:], mean=mean)

    # 기준 log π_ref 는 미리 한 번 (어댑터 끔)
    def refs(rows: List[Tuple[List[int], int, List[int], int]]) -> List[Tuple[float, float]]:
        out_ = []
        with pol.reference():
            for c, nc, j, nj in rows:
                out_.append((seq(c, nc).item(), seq(j, nj).item()))
        return out_

    pol.model.eval()
    ref, vref = refs(data), refs(val)
    pol.model.train()
    params = pol.params
    opt = torch.optim.AdamW(params, lr=lr)
    log: Dict[str, Any] = {"base": base, "pairs": len(data), "val_pairs": len(val), "beta": beta, "lr": lr, "epochs": epochs,
                           "nll_coef": nll_coef, "steps": []}
    t0 = time.time()
    idx = list(range(len(data)))
    for ep in range(epochs):
        random.shuffle(idx)
        opt.zero_grad()
        acc, losses, nlls = 0, [], []
        for k, i in enumerate(idx):
            c, nc, j, nj = data[i]
            rc, rj = ref[i]
            lc = seq(c, nc)
            lj = seq(j, nj)
            margin = beta * ((lc - rc) - (lj - rj))
            loss = -torch.nn.functional.logsigmoid(margin)
            if nll_coef > 0:
                nll = -seq(c, nc, mean=True)
                loss = loss + nll_coef * nll
                nlls.append(nll.item())
            (loss / grad_accum).backward()
            losses.append(loss.item())
            acc += int(margin.item() > 0)
            if (k + 1) % grad_accum == 0 or k == len(idx) - 1:
                torch.nn.utils.clip_grad_norm_(params, 1.0)
                opt.step()
                opt.zero_grad()
        rec: Dict[str, Any] = {"epoch": ep, "loss": round(sum(losses) / len(losses), 4), "pref_acc": round(acc / len(idx), 3),
                               "chosen_nll": round(sum(nlls) / len(nlls), 4) if nlls else None,
                               "seconds": round(time.time() - t0, 1)}
        if val:
            pol.model.eval()
            with torch.no_grad():
                ms = [beta * ((seq(c, nc).item() - vr[0]) - (seq(j, nj).item() - vr[1])) for (c, nc, j, nj), vr in zip(val, vref)]
            pol.model.train()
            rec.update(val_pref_acc=round(sum(m > 0 for m in ms) / len(ms), 3), val_margin=round(sum(ms) / len(ms), 4))
        log["steps"].append(rec)
        print(json.dumps(rec), flush=True)
        pol.save(str(out / ("adapter-ep%d" % ep)))
    pol.save(str(out / "adapter"))
    (out / "dpo_log.json").write_text(json.dumps(log, ensure_ascii=False, indent=2), encoding="utf-8")
    return out / "adapter"


__all__ = ["train_dpo"]


if __name__ == "__main__":   # 간단 실행: python -m btqa_distill.dpo <base> <pairs.jsonl> <out_dir>
    import sys
    print(train_dpo(sys.argv[1], sys.argv[2], sys.argv[3]))
