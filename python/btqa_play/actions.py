"""단계 형식 — "지금 상태 → 다음 행동 1개" (대화 기록 없음). 작은 로컬 LLM 실행·증류 데이터의 공통 형식.

한 단계 = 메시지 2개:
    system : 시스템 프롬프트(+목표) + 도구 목록 + 출력 형식     ← 판 내내 같은 글자 (CPU 추론에서 앞부분 캐시 재사용)
    user   : {"last_action": 직전 행동과 결과 | null, "observation": 지금 관찰}
             (history=K 이면 맨 앞에 "recent_actions": 그 이전 행동 최대 K개 요약, 오래된 것부터)
출력 (assistant) = JSON 한 개:  {"thought": "...", "tool": "<도구 이름>", "input": {...}}

- 직전 행동 1개는 자세히 넣는다: 실패 reason(not_visible 등)을 모르면 같은 실패를 되풀이하기 때문.
- history=K (선택, 기본 0 = 넣지 않음): 그 이전 행동 K개를 요약(tool·input·status·reason)으로 넣는다. 직전 하나만 보면
  A·B 를 번갈아 하는 반복이나 같은 행동의 N번째 반복을 알 수 없다 (AI_QA r13: 등록 버튼 20번, 약초↔포털 왕복).
  학습 형식과 실행 형식의 K 가 같아야 한다 — 기록(StepRecorder)에 history 를 남기고, 옛 기록은 dataset.add_history 로 다시 만든다.
- action_schema(tools) = 출력 JSON Schema (도구마다 input 스키마가 다른 anyOf) → 서버가 문법으로 강제 (Ollama·llama.cpp)
- LlmPilot(stateless)·JsonActionBackend·PilotRelay(기록)·StepRecorder 가 같은 함수를 쓴다 → 학습 형식 == 실행 형식
"""
from __future__ import annotations

import json
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

ACTION_FORMAT = """출력 형식: JSON 객체 하나만 출력한다. 다른 글은 쓰지 않는다.
{"thought": "왜 이 행동을 하는지 한 문장", "tool": "도구 이름", "input": {도구 인자}}
- 한 번에 도구 하나. 인자는 도구 목록의 이름만 쓰고, 쓰지 않는 선택 인자는 null.
- 입력의 last_action 은 직전 행동과 그 결과다 (status·reason). 실패했으면 방법을 바꾼다."""


def _tool_line(t: Mapping[str, Any]) -> str:
    props = (t.get("input_schema") or {}).get("properties") or {}
    args = ", ".join("%s%s" % (k, "" if _required(t, k) else "?") for k in props)
    return "- %s(%s): %s" % (t["name"], args, t.get("description", ""))


def _required(t: Mapping[str, Any], k: str) -> bool:
    p = ((t.get("input_schema") or {}).get("properties") or {}).get(k) or {}
    return not any(isinstance(b, Mapping) and b.get("type") == "null" for b in p.get("anyOf") or [])


def with_goal(system: str, goal: Optional[str]) -> str:
    """시스템 프롬프트 + 목표 (단계 형식에서는 목표도 고정 앞부분에 둔다)."""
    return system.rstrip() + ("\n\n목표: " + goal.strip() if goal else "")


def action_system_prompt(system: str, tools: Sequence[Mapping[str, Any]]) -> str:
    """단계 형식의 system 메시지 (판 내내 고정) = system(+목표) + 도구 목록 + 출력 형식."""
    tool_list = "도구 (? = 선택 인자, 안 쓰면 null):\n" + "\n".join(_tool_line(t) for t in tools)
    return "\n\n".join([system.rstrip(), tool_list, ACTION_FORMAT])


def action_schema(tools: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    """출력 JSON Schema — 도구마다 {thought, tool(enum 1개), input(그 도구 스키마)} 의 anyOf."""
    variants = []
    for t in tools:
        s = dict(t.get("input_schema") or {"type": "object", "properties": {}})
        s.pop("strict", None)
        variants.append({"type": "object",
                         "properties": {"thought": {"type": "string"}, "tool": {"type": "string", "enum": [t["name"]]},
                                        "input": s},
                         "required": ["thought", "tool", "input"], "additionalProperties": False})
    return {"anyOf": variants}


def step_input(observation: Mapping[str, Any], last_action: Optional[Mapping[str, Any]] = None,
               recent: Optional[Sequence[Mapping[str, Any]]] = None) -> str:
    """단계 형식의 user 메시지. observation 은 이미 줄인 것(compact_observation).
    recent: None = 기록 없음 (옛 형식 그대로), 리스트 = recent_actions (빈 리스트도 넣는다 — 판 첫 단계도 같은 모양)."""
    d: Dict[str, Any] = {}
    if recent is not None:
        d["recent_actions"] = [dict(x) for x in recent]
    d["last_action"] = dict(last_action) if last_action else None
    d["observation"] = observation
    return json.dumps(d, ensure_ascii=False, separators=(",", ":"))


def history_entry(last_action: Mapping[str, Any]) -> Dict[str, Any]:
    """last_action_record → recent_actions 한 칸 (짧게: tool·input·status·reason, 오류는 앞 80자, 전투는 cleared 만)."""
    e: Dict[str, Any] = {"tool": last_action.get("tool"), "input": dict(last_action.get("input") or {})}
    for k in ("status", "reason"):
        if last_action.get(k) is not None:
            e[k] = last_action[k]
    if last_action.get("error"):
        e["error"] = str(last_action["error"])[:80]
    if last_action.get("combats"):
        e["cleared"] = all(bool(c.get("cleared")) for c in last_action["combats"])
    return e


class ActionHistory:
    """판 하나의 행동 기록 → step_input 의 (last_action, recent). k=0 이면 recent=None (옛 형식)."""

    def __init__(self, k: int = 0):
        if k < 0:
            raise ValueError("history 는 0 이상")
        self.k = int(k)
        self._items: List[Dict[str, Any]] = []
        self.last: Optional[Dict[str, Any]] = None

    def push(self, last_action: Mapping[str, Any]) -> None:
        if self.last is not None and self.k:
            self._items.append(history_entry(self.last))
            del self._items[:-self.k]
        self.last = dict(last_action)

    def recent(self) -> Optional[List[Dict[str, Any]]]:
        return list(self._items) if self.k else None

    def user(self, observation: Mapping[str, Any]) -> str:
        return step_input(observation, self.last, self.recent())


def last_action_record(tool: str, tool_input: Mapping[str, Any], out: Mapping[str, Any]) -> Dict[str, Any]:
    """도구 결과(GameTools.execute 출력) → 다음 단계 입력의 last_action (관찰·전투 세부는 빼고 결과만)."""
    rec: Dict[str, Any] = {"tool": tool, "input": {k: v for k, v in tool_input.items() if v is not None}}
    for k in ("status", "reason", "reason_detail", "interrupted_by", "error", "message"):
        if out.get(k) is not None:
            rec[k] = out[k]
    if out.get("combats"):
        rec["combats"] = [{k: c.get(k) for k in ("enemy", "cleared", "deaths", "potions_used")} for c in out["combats"]]
    return rec


def action_text(thought: str, tool: str, tool_input: Mapping[str, Any]) -> str:
    """assistant 출력 (학습 목표) — 키 순서 고정."""
    return json.dumps({"thought": thought, "tool": tool, "input": dict(tool_input)}, ensure_ascii=False, separators=(",", ":"))


def parse_action(text: str, tool_names: Sequence[str]) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    """모델 출력 → ({thought, tool, input}, None) 또는 (None, 오류). 앞뒤 잡글·코드펜스는 걷어낸다."""
    s = (text or "").strip()
    if s.startswith("```"):
        s = s.strip("`")
        s = s[s.find("{"):] if "{" in s else s
    i, j = s.find("{"), s.rfind("}")
    if i < 0 or j <= i:
        return None, "JSON 객체가 없음"
    try:
        d = json.loads(s[i:j + 1])
    except ValueError as e:
        return None, "JSON 오류: %s" % e
    if not isinstance(d, dict):
        return None, "JSON 객체가 아님"
    tool = d.get("tool")
    if tool not in tool_names:
        return None, "모르는 도구 %r (도구: %s)" % (tool, ", ".join(tool_names))
    inp = d.get("input") if isinstance(d.get("input"), dict) else {}
    return {"thought": str(d.get("thought") or ""), "tool": tool, "input": inp}, None


__all__ = ["ACTION_FORMAT", "with_goal", "action_system_prompt", "action_schema", "step_input", "last_action_record", "action_text",
           "history_entry", "ActionHistory",
           "parse_action"]
