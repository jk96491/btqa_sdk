"""교사 라벨 검토 — 증류 데이터의 정답(교사·승인한 학생 행동)을 자동으로 표시하고, 사람·에이전트가 확인한다. 표준 라이브러리만.

자동 표시 (relay 가 기록할 때, label_flags):
    label_failed         정답으로 기록한 행동을 그대로 실행했는데 결과가 failed·error
                         (AI_QA 2026-11-25: 몰아서 승인하다 게임에서 실패한 학생 행동 13개가 정답으로 들어갔다)
    repeat_no_progress   직전 단계와 같은 행동이 또 성공했는데 진척 서명이 그대로 (반복 — 진척 없는 성공)
교사 표시 (step --unsure):
    uncertain            교사가 확신하지 못한 라벨. 생각 문장이 아니라 **필드**로 남긴다 (글자로 넣으면 학생이 따라 쓴다)

해소:
    reviewed  "<메모>"     확인해 보니 맞다 → 학습에 쓴다
    edited    "<이유>"     정답을 고쳤다 (target·target_text 교체) → 교정 단계로 센다
dataset 은 해소되지 않은 needs_review 를 기본으로 뺀다 (dataset.py).

"진척"이 무엇인지는 게임이 정한다 (progress_keys, PLAY_PROTOCOL 4.2.1). SDK 는 서명이 바뀌었는지만 본다.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Union

from btqa_play.actions import action_text
from btqa_play.record import read_steps, write_steps

FAILED_STATUSES = ("failed",)
FLAGS = ("label_failed", "repeat_no_progress")


def _act(tool: str, inp: Mapping[str, Any]) -> str:
    return json.dumps([tool, {k: v for k, v in inp.items() if v is not None}], sort_keys=True, ensure_ascii=False)


def label_flags(label: Sequence[Any], executed: Sequence[Any], result: Mapping[str, Any],
                prev_executed: Optional[Sequence[Any]] = None, prev_ok: bool = False,
                progressed: Optional[bool] = None) -> List[str]:
    """label/executed = (tool, input). result = last_action_record (status·reason·error).
    prev_executed·prev_ok = 직전 단계에 실행한 행동과 그 성공 여부, progressed = 이번 행동 뒤 진척 서명이 바뀌었는가."""
    out: List[str] = []
    same = _act(label[0], label[1]) == _act(executed[0], executed[1])
    ok = not result.get("error") and result.get("status") not in FAILED_STATUSES
    if same and not ok:
        out.append("label_failed")
    if (same and ok and prev_ok and prev_executed is not None and progressed is False
            and _act(prev_executed[0], prev_executed[1]) == _act(executed[0], executed[1])):
        out.append("repeat_no_progress")
    return out


def is_resolved(rec: Mapping[str, Any]) -> bool:
    return bool(rec.get("reviewed") or rec.get("edited"))


def pending(rec: Mapping[str, Any], include_uncertain: bool = True) -> List[str]:
    """해소되지 않은 검토 사유."""
    if is_resolved(rec):
        return []
    out = list(rec.get("needs_review") or [])
    if include_uncertain and rec.get("uncertain"):
        out.append("uncertain")
    return out


def review_list(paths: Sequence[Union[str, Path]], include_uncertain: bool = True) -> List[Dict[str, Any]]:
    """검토할 줄 목록: {file, run, step, why, tool, input, thought, result}."""
    out = []
    for p in paths:
        for r in read_steps(p):
            why = pending(r, include_uncertain)
            if why:
                t = r.get("target") or {}
                out.append({"file": str(p), "run": r.get("run"), "step": r.get("step"), "why": why, "tool": t.get("tool"),
                            "input": t.get("input"), "thought": t.get("thought"), "result": r.get("result")})
    return out


def resolve(path: Union[str, Path], run: str, step: int, note: str, tool: Optional[str] = None,
            tool_input: Optional[Mapping[str, Any]] = None, thought: Optional[str] = None) -> Dict[str, Any]:
    """한 줄 해소. tool 이 없으면 reviewed=note (라벨이 맞음). tool 이 있으면 정답을 고치고 edited=note.
    run 은 끝부분 일치 (예 '-s5'), 한 줄만 맞아야 한다."""
    recs = read_steps(path)
    hit = [i for i, r in enumerate(recs) if int(r.get("step") or 0) == int(step)
           and (str(r.get("run")) == run or str(r.get("run")).endswith(run))]
    if len(hit) != 1:
        raise ValueError("run=%r step=%s 에 맞는 줄이 %d개 (1개여야 한다)" % (run, step, len(hit)))
    r = recs[hit[0]]
    if not note:
        raise ValueError("메모(이유)가 필요하다")
    if tool is None:
        r["reviewed"] = note
    else:
        t = r.get("target") or {}
        new = {"thought": t.get("thought", "") if thought is None else thought, "tool": tool,
               "input": dict(tool_input if tool_input is not None else {})}
        r.setdefault("original_target", dict(t))
        r["target"] = new
        r["target_text"] = action_text(new["thought"], new["tool"], new["input"])
        r["edited"] = note
    write_steps(path, recs)
    return r


__all__ = ["FLAGS", "label_flags", "is_resolved", "pending", "review_list", "resolve"]
