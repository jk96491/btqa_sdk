"""단계 기록(record.py, btqa-play-step/1) → 학습용 대화 데이터 (SFT). 표준 라이브러리만.

    stats = build_dataset(["teacher.jsonl", "corrections.jsonl"], "distill/data/v1", val_ratio=0.1)

출력 (out_dir):
    train.jsonl / val.jsonl  한 줄 = {"messages": [system, user, assistant(target_text)], "meta": {run, step, source}}
    stats.json               읽은 줄·뺀 이유별 수·중복·판(run) 수·평균 길이

거르기 (기본):
    - source 가 sources 에 없는 줄 (기본 teacher 만 — model 기록은 평가용, 교정 데이터는 source=teacher 로 기록)
    - grounding 이 비어 있지 않은 줄 (학생 입력에 근거가 없는 판단 — 배울 수 없다)
    - 출력 형식 오류·모르는 도구 (target.tool 이 비었거나 result.error 가 형식 오류)
    - 중복: 같은 (system, user) → 같은 target 은 하나만. 같은 입력에 다른 target 이 있으면 모두 남기고 stats 에 센다
나누기: **판(run) 단위**로 val 을 뗀다 (같은 판의 이웃 단계가 train·val 에 섞이면 val 이 낙관적이 된다).
"""
from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple, Union

from btqa_play.record import STEP_FORMAT, read_steps

DATASET_FORMAT = "btqa-play-sft/1"


def _key(*parts: str) -> str:
    h = hashlib.sha1()
    for p in parts:
        h.update(p.encode("utf-8"))
        h.update(b"\x00")
    return h.hexdigest()


def _reason_to_drop(rec: Dict[str, Any], sources: Sequence[str], keep_ungrounded: bool) -> str:
    if rec.get("format") != STEP_FORMAT:
        return "format"
    if rec.get("source") not in sources:
        return "source:%s" % rec.get("source")
    t = rec.get("target") or {}
    if not t.get("tool"):
        return "invalid_output"
    err = str((rec.get("result") or {}).get("error") or "")
    if err.startswith("출력 형식 오류") or err.startswith("모르는 도구"):
        return "invalid_output"
    if rec.get("grounding") and not keep_ungrounded:
        return "ungrounded"
    if not rec.get("target_text") or len(rec.get("messages") or []) != 2:
        return "incomplete"
    return ""


def to_sample(rec: Dict[str, Any]) -> Dict[str, Any]:
    msgs = [dict(m) for m in rec["messages"]] + [{"role": "assistant", "content": rec["target_text"]}]
    return {"format": DATASET_FORMAT, "messages": msgs,
            "meta": {"run": rec.get("run"), "step": rec.get("step"), "source": rec.get("source"), "tool": rec["target"]["tool"]}}


def build_dataset(inputs: Sequence[Union[str, Path]], out_dir: Union[str, Path], val_ratio: float = 0.1, seed: int = 0,
                  sources: Sequence[str] = ("teacher",), keep_ungrounded: bool = False) -> Dict[str, Any]:
    recs: List[Dict[str, Any]] = []
    for p in inputs:
        recs += read_steps(p)
    dropped: Dict[str, int] = {}
    seen: Dict[str, str] = {}
    conflicts = 0
    kept: List[Dict[str, Any]] = []
    for r in recs:
        why = _reason_to_drop(r, sources, keep_ungrounded)
        if why:
            dropped[why] = dropped.get(why, 0) + 1
            continue
        k_in = _key(r["messages"][0]["content"], r["messages"][1]["content"])
        k_out = _key(r["target_text"])
        if k_in in seen:
            if seen[k_in] == k_out:
                dropped["duplicate"] = dropped.get("duplicate", 0) + 1
                continue
            conflicts += 1
        seen[k_in] = k_out
        kept.append(r)

    runs = sorted({str(r.get("run")) for r in kept})
    rng = random.Random(seed)
    rng.shuffle(runs)
    n_val = max(1, int(round(len(runs) * val_ratio))) if len(runs) > 1 and val_ratio > 0 else 0
    val_runs = set(runs[:n_val])
    train = [to_sample(r) for r in kept if str(r.get("run")) not in val_runs]
    val = [to_sample(r) for r in kept if str(r.get("run")) in val_runs]

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for name, rows in (("train.jsonl", train), ("val.jsonl", val)):
        with (out / name).open("w", encoding="utf-8") as f:
            for s in rows:
                f.write(json.dumps(s, ensure_ascii=False) + "\n")

    def _mean(rows: Iterable[Dict[str, Any]], i: int) -> float:
        rows = list(rows)
        return round(sum(len(s["messages"][i]["content"]) for s in rows) / max(1, len(rows)), 1)

    tools: Dict[str, int] = {}
    for s in train + val:
        tools[s["meta"]["tool"]] = tools.get(s["meta"]["tool"], 0) + 1
    stats = {"format": DATASET_FORMAT, "inputs": [str(p) for p in inputs], "read": len(recs), "kept": len(kept),
             "dropped": dropped, "conflicting_inputs": conflicts, "runs": len(runs), "val_runs": sorted(val_runs),
             "train": len(train), "val": len(val), "tools": tools,
             "mean_chars": {"user": _mean(train + val, 1), "assistant": _mean(train + val, 2)},
             "distinct_systems": len({s["messages"][0]["content"] for s in train + val})}
    (out / "stats.json").write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
    return stats


def read_samples(path: Union[str, Path]) -> List[Dict[str, Any]]:
    return [json.loads(x) for x in Path(path).read_text(encoding="utf-8").splitlines() if x.strip()]


def split_prompt(sample: Dict[str, Any]) -> Tuple[List[Dict[str, str]], str]:
    """(프롬프트 메시지 [system, user], 목표 assistant 글) — 학습기가 loss 를 assistant 에만 걸 때 쓴다."""
    m = sample["messages"]
    return [dict(x) for x in m[:-1]], m[-1]["content"]


__all__ = ["DATASET_FORMAT", "build_dataset", "to_sample", "read_samples", "split_prompt"]
