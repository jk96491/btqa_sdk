"""단계 기록기 — 증류(교사 → 작은 모델) 데이터 / 모델 평가 기록. 한 단계 = JSONL 한 줄.

한 줄 (format "btqa-play-step/1"):
    {"format", "run", "step", "source": "teacher"|"model", "model",
     "messages": [{"role":"system","content":...}, {"role":"user","content":...}],   ← 학생이 받는 입력 그대로 (actions.py)
     "target": {"thought","tool","input"},  "target_text": "<assistant 출력 JSON>",   ← 학습 목표
     "result": {status, reason, ...},  "usage": {input_tokens, output_tokens},  "latency_s",
     "grounding": ["…"]}                                                           ← 입력에 근거 없는 행동 (빈 목록 = 통과)

grounding 검사: 교사가 학생이 볼 수 없는 정보(대화 기록·기억)로 판단한 샘플을 골라내기 위한 것.
    - 행동 인자의 문자열 id(target·ui_id·item_id 등)가 그 단계 입력(user 메시지)에 없으면 "id_not_in_input:<값>"
    - 좌표(pos)가 관찰의 어느 위치(player·entities)와도 radius 이상 떨어져 있으면 "pos_unsupported:<좌표>"
    학습 데이터로 만들 때 grounding 이 비어 있지 않은 샘플은 빼거나 검토한다.
"""
from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Union

from btqa_play.actions import action_text

STEP_FORMAT = "btqa-play-step/1"
#: id 로 검사할 인자 키 (프로토콜 행동 인자 이름 규칙: 엔티티·UI·아이템 id)
ID_ARG_KEYS = ("target", "ui_id", "item_id", "root")


def _positions(obs: Mapping[str, Any]) -> Iterable[List[float]]:
    p = (obs.get("player") or {}).get("pos")
    if isinstance(p, list):
        yield p
    for e in obs.get("entities") or []:
        if isinstance(e, Mapping) and isinstance(e.get("pos"), list):
            yield e["pos"]


def grounding_issues(user_text: str, observation: Mapping[str, Any], tool_input: Mapping[str, Any],
                     radius: float = 6.0) -> List[str]:
    out: List[str] = []
    for k in ID_ARG_KEYS:
        v = tool_input.get(k)
        if isinstance(v, str) and v and ('"%s"' % v) not in user_text:
            out.append("id_not_in_input:%s" % v)
    pos = tool_input.get("pos")
    if isinstance(pos, list) and pos and all(isinstance(x, (int, float)) for x in pos):
        near = [math.dist(pos[:len(q)], q[:len(pos)]) for q in _positions(observation)]   # type: ignore[attr-defined]
        if not near or min(near) > radius:
            out.append("pos_unsupported:%s" % pos)
    return out


class StepRecorder:
    """JSONL 에 한 단계씩 덧붙인다. run = 판 이름 (같은 파일에 여러 판)."""

    def __init__(self, path: Union[str, Path], run: str, source: str = "teacher", model: str = "") -> None:
        if source not in ("teacher", "model"):
            raise ValueError("source 는 teacher 또는 model")
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.run, self.source, self.model = run, source, model
        self.step = 0
        self.flagged = 0

    def write(self, system: str, user: str, observation: Mapping[str, Any], thought: str, tool: str,
              tool_input: Mapping[str, Any], result: Mapping[str, Any], usage: Optional[Mapping[str, int]] = None,
              latency_s: Optional[float] = None, extra: Optional[Mapping[str, Any]] = None) -> Dict[str, Any]:
        self.step += 1
        g = grounding_issues(user, observation, tool_input)
        self.flagged += bool(g)
        rec = {"format": STEP_FORMAT, "run": self.run, "step": self.step, "source": self.source, "model": self.model,
               "time": time.strftime("%Y-%m-%dT%H:%M:%S"),
               "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
               "target": {"thought": thought, "tool": tool, "input": dict(tool_input)},
               "target_text": action_text(thought, tool, tool_input),
               "result": dict(result), "usage": dict(usage or {}), "latency_s": latency_s, "grounding": g}
        if extra:
            rec.update({k: v for k, v in extra.items() if k not in rec})   # 예 student = 교정 모드 학생 제안 (DPO 의 rejected)
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        return rec


def read_steps(path: Union[str, Path]) -> List[Dict[str, Any]]:
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


__all__ = ["STEP_FORMAT", "StepRecorder", "grounding_issues", "read_steps"]
