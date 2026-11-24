"""게임 행동 ↔ LLM 도구 정의 계층 (LlmPilot·PilotRelay·이후 MCP 래퍼가 공유).

SDK 는 행동의 **내용**(설명 문장·인자 의미)을 들고 있지 않다. 받은 선언을 strict 도구로 바꿀 뿐이다.

- build_tools(description, spec) : play_description.actions 중 **게임이 지원한다고 한 것만** 도구로 (+ 메타 도구)
    행동마다 설명·스키마 출처 (앞의 것이 우선, 설명·스키마 따로):
      game    — play_description.actions[].description / schema (PLAY_PROTOCOL 4.2.1)
      spec    — 외부 플레이 스펙 actions.<이름> (spec.py, 게임이 아직 선언하지 않을 때)
      generic — args 이름마다 nullable 문자열, 설명 = 이름과 args 요약  ← validate 가 경고
- tool_sources(description, spec) : 행동마다 "설명출처/스키마출처"
- validate_description(description, spec) : 게임 선언 규칙 위반 + (경고) 설명·스키마가 어디에도 없는 행동
- GameTools.execute(name, input) : 도구 호출 → PlayClient → LLM 에게 줄 결과(dict, 프로토콜 4.6 필드)
- compact_observation(obs) : LLM 에게 보일 관찰 (seq 제거·숫자 반올림)

도구 스키마는 strict 용: 모든 속성 required + additionalProperties false, 선택 인자는 null 허용.
메타 도구(inspect_ui·finish)는 클라이언트(SDK) 도구라 스키마는 SDK 가 정하고, 설명은 스펙으로 바꿀 수 있다.
"""
from __future__ import annotations

import json
from typing import Any, Callable, Dict, List, Mapping, Optional, Tuple

from btqa_play.client import ActResult, Combat, PlayClient, PlayError
from btqa_play.spec import (META_TOOL_NAMES, PlaySpec, check_guide, check_progress_keys, check_schema)

NULLABLE_STR = {"anyOf": [{"type": "string"}, {"type": "null"}]}

#: 클라이언트 메타 도구 — 스키마는 SDK 계약, 설명은 기본 문구(게임 개념 없음) 또는 spec.meta_tools
META_TOOLS: List[Dict[str, Any]] = [
    {
        "name": "inspect_ui",
        "description": "UI 계층 트리를 본다 (디버그용). root = UI 요소 id 또는 null(전체).",
        "input_schema": {"type": "object", "properties": {"root": NULLABLE_STR}, "required": ["root"], "additionalProperties": False},
    },
    {
        "name": "finish",
        "description": "플레이를 끝낸다. 목표를 이뤘거나 더 진행할 수 없을 때. blocked 에 막힌 이유를 적는다.",
        "input_schema": {
            "type": "object",
            "properties": {"success": {"type": "boolean"}, "summary": {"type": "string"}, "blocked": NULLABLE_STR},
            "required": ["success", "summary", "blocked"], "additionalProperties": False,
        },
    },
]


def _is_nullable(prop: Mapping[str, Any]) -> bool:
    t = prop.get("type")
    if t == "null" or (isinstance(t, list) and "null" in t):
        return True
    return any(isinstance(b, Mapping) and b.get("type") == "null" for b in prop.get("anyOf") or [])


def _nullable(prop: Mapping[str, Any]) -> Dict[str, Any]:
    """선택 인자 → strict 스키마용 'null 허용' 형태 (설명은 바깥에 둔다)."""
    if _is_nullable(prop):
        return dict(prop)
    inner = {k: v for k, v in prop.items() if k != "description"}
    out: Dict[str, Any] = {"anyOf": [inner, {"type": "null"}]}
    if "description" in prop:
        out["description"] = prop["description"]
    return out


def _from_schema(schema: Mapping[str, Any]) -> Dict[str, Any]:
    """선언된 JSON Schema(object) → strict 속성 (required 가 아닌 것은 null 허용, required 생략 = 모두 필수)."""
    props = dict(schema.get("properties") or {})
    required = schema.get("required")
    req = set(props) if required is None else set(required)
    return {k: (dict(v) if k in req else _nullable(v)) for k, v in props.items()}


def _has_schema(s: Any) -> bool:
    return isinstance(s, Mapping) and isinstance(s.get("properties"), Mapping)


def _has_text(s: Any) -> bool:
    return isinstance(s, str) and bool(s.strip())


def _action_tool(a: Mapping[str, Any], spec: PlaySpec) -> Tuple[Dict[str, Any], str]:
    name = str(a["name"])
    sa = spec.actions.get(name) or {}
    if _has_schema(a.get("schema")):
        props, s_src = _from_schema(a["schema"]), "game"
    elif _has_schema(sa.get("schema")):
        props, s_src = _from_schema(sa["schema"]), "spec"
    else:
        props, s_src = {k: dict(NULLABLE_STR) for k in (a.get("args") or {})}, "generic"
    if _has_text(a.get("description")):
        desc, d_src = a["description"], "game"
    elif _has_text(sa.get("description")):
        desc, d_src = sa["description"], "spec"
    else:
        desc, d_src = "%s (인자 %s)" % (name, json.dumps(a.get("args") or {}, ensure_ascii=False)), "generic"
    tool = {"name": name, "description": desc,
            "input_schema": {"type": "object", "properties": props, "required": list(props), "additionalProperties": False}}
    return tool, "%s/%s" % (d_src, s_src)


def _meta_tools(spec: PlaySpec) -> List[Dict[str, Any]]:
    out = []
    for t in META_TOOLS:
        d = (spec.meta_tools.get(t["name"]) or {}).get("description")
        out.append(dict(t, description=d) if _has_text(d) else dict(t))
    return out


def build_tools(description: Mapping[str, Any], spec: Optional[PlaySpec] = None, strict: bool = True) -> List[Dict[str, Any]]:
    """play_description.actions → LLM 도구 목록 (게임이 준 순서 그대로, 메타 도구는 뒤). 형식 = 중립
    {name, description, input_schema[, strict]} — 백엔드가 자기 API 형식으로 바꾼다."""
    spec = spec or PlaySpec()
    tools = [_action_tool(a, spec)[0] for a in description.get("actions") or [] if isinstance(a, Mapping) and a.get("name")]
    tools += _meta_tools(spec)
    if strict:
        tools = [dict(t, strict=True) for t in tools]
    return tools


def tool_sources(description: Mapping[str, Any], spec: Optional[PlaySpec] = None) -> Dict[str, str]:
    """행동 이름 → "설명출처/스키마출처" (game | spec | generic)."""
    spec = spec or PlaySpec()
    return {str(a["name"]): _action_tool(a, spec)[1] for a in description.get("actions") or [] if isinstance(a, Mapping) and a.get("name")}


def validate_description(description: Mapping[str, Any], spec: Optional[PlaySpec] = None,
                         require_text: bool = True) -> List[str]:
    """게임 선언(PLAY_PROTOCOL 4.2.1) 검사. 빈 목록 = 통과.

    require_text=True 면 게임에도 외부 스펙에도 설명·스키마가 없는 행동(일반형으로 떨어지는 것)도 문제로 보고한다.
    """
    spec = spec or PlaySpec()
    out: List[str] = []
    actions = description.get("actions")
    if not isinstance(actions, list) or not actions:
        return ["actions 가 비었거나 목록이 아님"]
    seen = set()
    for i, a in enumerate(actions):
        if not isinstance(a, Mapping) or not isinstance(a.get("name"), str) or not a["name"]:
            out.append("actions[%d]: name 없음" % i)
            continue
        n = a["name"]
        if n in seen:
            out.append("%s: 이름 중복" % n)
        seen.add(n)
        if n in META_TOOL_NAMES:
            out.append("%s: 클라이언트 메타 도구와 같은 이름" % n)
        if "description" in a and not _has_text(a["description"]):
            out.append("%s: description 은 비어 있지 않은 문자열" % n)
        args = a.get("args")
        names = list(args) if isinstance(args, Mapping) else None
        if "schema" in a:
            out += check_schema(n, a["schema"], names)
        elif _has_schema((spec.actions.get(n) or {}).get("schema")) and names is not None:
            out += check_schema("%s (스펙)" % n, spec.actions[n]["schema"], names)
        if require_text:
            src = _action_tool(a, spec)[1]
            if "generic" in src:
                out.append("%s: 설명·스키마가 게임 선언에도 외부 스펙에도 없음 (출처 %s)" % (n, src))
    for n in sorted(set(spec.actions) - seen):
        out.append("스펙 actions.%s: 게임이 지원하지 않는 행동 (무시됨)" % n)
    if "pilot_guide" in description:
        out += check_guide("pilot_guide", description["pilot_guide"])
    if "progress_keys" in description:
        out += check_progress_keys("progress_keys", description["progress_keys"])
    return out


def _round(x: Any) -> Any:
    if isinstance(x, float):
        return round(x, 2)
    if isinstance(x, list):
        return [_round(v) for v in x]
    if isinstance(x, dict):
        return {k: _round(v) for k, v in x.items()}
    return x


def compact_observation(obs: Mapping[str, Any]) -> Dict[str, Any]:
    o = {k: v for k, v in obs.items() if k not in ("type", "seq")}
    return _round(o)


class GameTools:
    """도구 호출 실행기. responder = 전투 트리거 응답기 (trigger → params, on_combat_started 선택)."""

    def __init__(self, client: PlayClient, responder: Optional[Callable[..., Any]] = None, inspect_depth: int = 4) -> None:
        self.client = client
        self.responder = responder
        self.inspect_depth = inspect_depth
        self.finished: Optional[Dict[str, Any]] = None
        self.combats: List[Combat] = []
        self.last_observation: Dict[str, Any] = {}
        self.actions = 0

    def execute(self, name: str, tool_input: Mapping[str, Any]) -> Dict[str, Any]:
        if name == "finish":
            self.finished = dict(tool_input)
            return {"ok": True}
        if name == "inspect_ui":
            root = tool_input.get("root")
            kw = {"root": root} if root else {}
            return _round(self.client.observe("hierarchy", max_depth=self.inspect_depth, **kw))
        args = {k: v for k, v in tool_input.items() if v is not None}
        self.actions += 1
        r: ActResult = self.client.act(name, args, responder=self.responder)
        self.combats += r.combats
        if r.is_error:   # 인자·행동 이름 오류 (action_result 아님)
            return {"error": r.get("code"), "message": r.get("message")}
        self.last_observation = r.observation
        out: Dict[str, Any] = {"status": r.status}
        for k in ("reason", "reason_detail", "interrupted_by", "game_seconds"):   # PLAY_PROTOCOL 4.6
            if r.get(k) is not None:
                out[k] = _round(r.get(k))
        if r.combats:
            out["combats"] = [c.summary() for c in r.combats]
        out["observation"] = compact_observation(r.observation)
        return out


def to_text(result: Mapping[str, Any]) -> str:
    return json.dumps(result, ensure_ascii=False, separators=(",", ":"))


__all__ = ["META_TOOLS", "build_tools", "tool_sources", "validate_description",
           "GameTools", "compact_observation", "to_text", "PlayError"]
