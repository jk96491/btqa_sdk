"""단계 기록기 — 증류(교사 → 작은 모델) 데이터 / 모델 평가 기록. 한 단계 = JSONL 한 줄.

한 줄 (format "btqa-play-step/1"):
    {"format", "run", "step", "source": "teacher"|"model", "model",
     "messages": [{"role":"system","content":...}, {"role":"user","content":...}],   ← 학생이 받는 입력 그대로 (actions.py)
     "target": {"thought","tool","input"},  "target_text": "<assistant 출력 JSON>",   ← 학습 목표
     "result": {status, reason, ...},  "usage": {input_tokens, output_tokens},  "latency_s",
     "grounding": ["…"],                                                           ← 입력에 근거 없는 행동 (빈 목록 = 통과)
     "episode": {"seed", "class", "options"},                                      ← 판 시작 조건 (분기 재생·on-policy 쌍용, 선택)
     "needs_review": ["label_failed", …], "uncertain": true, "edited": "<이유>", "reviewed": "<메모>"}   ← 라벨 검토 (review.py)

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

    def __init__(self, path: Union[str, Path], run: str, source: str = "teacher", model: str = "",
                 episode: Optional[Mapping[str, Any]] = None) -> None:
        if source not in ("teacher", "model"):
            raise ValueError("source 는 teacher 또는 model")
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.run, self.source, self.model = run, source, model
        self.episode = {k: v for k, v in dict(episode or {}).items() if v not in (None, {}, "")}
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
        if self.episode:
            rec["episode"] = dict(self.episode)
        if extra:
            rec.update({k: v for k, v in extra.items() if k not in rec})   # 예 student = 교정 모드 학생 제안 (DPO 의 rejected)
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        return rec


def with_history(records: Iterable[Mapping[str, Any]], k: int) -> List[Dict[str, Any]]:
    """기록의 user 글을 history=K 형식으로 다시 만든다 (actions.ActionHistory 와 같은 결과). k=0 이면 recent_actions 를 뗀다.

    판(run)마다 step 순서로, 앞 단계들 user 의 last_action(= 실제로 실행된 행동) 을 이어 붙인다. step 이 끊긴 곳에서는
    그 이전을 모르므로 기록을 비우고 다시 시작한다. 입력 순서는 지킨다 (판 안의 순서만 쓴다)."""
    from btqa_play.actions import ActionHistory, step_input
    recs = [dict(r) for r in records]
    order: Dict[str, List[int]] = {}
    for i, r in enumerate(recs):
        order.setdefault(str(r.get("run")), []).append(i)
    for idx in order.values():
        idx.sort(key=lambda i: int(recs[i].get("step") or 0))
        hist, prev_step = ActionHistory(k), None
        for i in idx:
            r = recs[i]
            msgs = r.get("messages") or []
            if len(msgs) != 2:
                continue
            try:
                d = json.loads(msgs[1]["content"])
            except (ValueError, TypeError):
                continue
            step = int(r.get("step") or 0)
            if prev_step is not None and step != prev_step + 1:
                hist = ActionHistory(k)
            prev_step = step
            la = d.get("last_action")
            if la is None:           # 판 첫 단계
                hist = ActionHistory(k)
            else:
                hist.push(la)
            text = step_input(d.get("observation") or {}, la, hist.recent())
            r["messages"] = [dict(msgs[0]), dict(msgs[1], content=text)]
    return recs


def read_steps(path: Union[str, Path]) -> List[Dict[str, Any]]:
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def write_steps(path: Union[str, Path], records: Iterable[Mapping[str, Any]]) -> None:
    """기록 전체를 다시 쓴다 (검토 결과 반영 등). 임시 파일에 쓰고 바꾼다."""
    p = Path(path)
    tmp = p.with_suffix(p.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    tmp.replace(p)


def episode_of(rec: Mapping[str, Any]) -> Dict[str, Any]:
    """기록 한 줄의 판 시작 조건. episode 필드가 없는 예전 기록은 run 이름 끝 '-s<seed>' 에서 seed 만."""
    import re
    ep = dict(rec.get("episode") or {})
    if "seed" not in ep:
        m = re.search(r"-s(\d+)$", str(rec.get("run") or ""))
        if m:
            ep["seed"] = int(m.group(1))
    return ep


__all__ = ["STEP_FORMAT", "StepRecorder", "grounding_issues", "read_steps", "write_steps", "episode_of", "with_history"]
