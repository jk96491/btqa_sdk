"""플레이 스펙 — LLM 플레이어에게 줄 **게임 고유 내용**(행동 설명·인자 스키마·안내·진척 기준)의 형식.

SDK 는 이 형식을 검사하고 합치고 변환할 뿐, 내용(문장·게임 개념)을 들고 있지 않다.
내용의 출처 (앞의 것이 우선, 키마다):
    1. 게임 선언  — play_description 의 actions[].description/schema, pilot_guide, progress_keys (PLAY_PROTOCOL 4.2.1)
    2. 외부 스펙  — 쓰는 쪽이 넘기는 같은 형식의 JSON 파일 (게임이 아직 선언하지 않을 때)
    3. 없음       — 일반형 도구(args 이름마다 nullable 문자열) + 검사 경고

형식 (docs/PLAY_SDK.md 2장):
    {
      "format": "btqa-play-spec/1",
      "actions": {"<행동 이름>": {"description": "...", "schema": {"type":"object","properties":{...},"required":[...]}}},
      "meta_tools": {"finish": {"description": "..."}, "inspect_ui": {"description": "..."}},
      "pilot_guide": {"system_prompt": "...", "about": "...", "rules": ["..."], "default_goal": "..."},
      "progress_keys": ["quests", "player.zone"],         # 진척 판단에 볼 관찰 경로 (점 구분)
      "observation_drop": ["screen.ui[role=text]"]        # LLM 에게 보이기 전에 뺄 관찰 경로 (입력 길이 — CPU 추론)
    }
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Union

SPEC_FORMAT = "btqa-play-spec/1"
SPEC_KEYS = ("format", "actions", "meta_tools", "pilot_guide", "progress_keys", "observation_drop")
GUIDE_KEYS = ("system_prompt", "about", "rules", "default_goal")
META_TOOL_NAMES = ("inspect_ui", "finish")
_JSON_TYPES = ("string", "number", "integer", "boolean", "array", "object", "null")


class SpecError(ValueError):
    """스펙 형식 오류."""


def check_schema(where: str, s: Any, arg_names: Optional[Sequence[str]] = None) -> List[str]:
    """인자 JSON Schema 검사 (게임 선언·외부 스펙 공통)."""
    if not isinstance(s, Mapping) or s.get("type") != "object" or not isinstance(s.get("properties"), Mapping):
        return ["%s: schema 는 {type:'object', properties:{...}}" % where]
    out: List[str] = []
    for k, p in s["properties"].items():
        if not isinstance(p, Mapping):
            out.append("%s.%s: 속성 스키마가 객체가 아님" % (where, k))
            continue
        t = p.get("type")
        types = t if isinstance(t, list) else [t] if t is not None else []
        if [x for x in types if x not in _JSON_TYPES] or (not types and "anyOf" not in p and "enum" not in p):
            out.append("%s.%s: type 이 JSON Schema 타입이 아님 (%s)" % (where, k, t))
    req = s.get("required")
    if req is not None and (not isinstance(req, list) or any(r not in s["properties"] for r in req)):
        out.append("%s: required 에 properties 에 없는 이름" % where)
    if arg_names is not None and set(arg_names) != set(s["properties"]):
        out.append("%s: args 키 %s ≠ schema 속성 %s" % (where, sorted(arg_names), sorted(s["properties"])))
    return out


def check_guide(where: str, g: Any) -> List[str]:
    if not isinstance(g, Mapping):
        return ["%s 는 객체" % where]
    out: List[str] = []
    for k in ("system_prompt", "about", "default_goal"):
        if k in g and not isinstance(g[k], str):
            out.append("%s.%s 는 문자열" % (where, k))
    if "rules" in g and not (isinstance(g["rules"], list) and all(isinstance(r, str) for r in g["rules"])):
        out.append("%s.rules 는 문자열 목록" % where)
    unknown = sorted(set(g) - set(GUIDE_KEYS))
    if unknown:
        out.append("%s: 모르는 키 %s" % (where, unknown))
    return out


def check_progress_keys(where: str, keys: Any) -> List[str]:
    if not (isinstance(keys, list) and keys and all(isinstance(k, str) and k for k in keys)):
        return ["%s 는 비어 있지 않은 문자열 목록 (관찰 경로, 예: \"player.zone\")" % where]
    return []


def check_observation_drop(where: str, paths: Any) -> List[str]:
    """관찰에서 뺄 경로: 점 구분 경로, 목록이면 원소마다 적용, 마지막 조각에 [키=값] 필터 가능 (예 screen.ui[role=text])."""
    if not (isinstance(paths, list) and all(isinstance(p, str) and p for p in paths)):
        return ["%s 는 문자열 목록 (관찰 경로, 예: \"hidden_included\", \"screen.ui[role=text]\")" % where]
    return ["%s: 필터 형식 [키=값] 이 아님: %s" % (where, p) for p in paths if "[" in p and not p.endswith("]") or p.count("[") > 1]


def validate_spec(d: Mapping[str, Any]) -> List[str]:
    """외부 스펙 dict 검사. 빈 목록 = 통과."""
    if not isinstance(d, Mapping):
        return ["스펙은 JSON 객체"]
    out: List[str] = []
    if d.get("format") != SPEC_FORMAT:
        out.append("format 은 \"%s\" (받은 값 %r)" % (SPEC_FORMAT, d.get("format")))
    out += ["모르는 키 %s" % k for k in sorted(set(d) - set(SPEC_KEYS)) if not k.startswith("_")]
    acts = d.get("actions", {})
    if not isinstance(acts, Mapping):
        out.append("actions 는 {이름: {description, schema}} 객체")
        acts = {}
    for n, a in acts.items():
        if n in META_TOOL_NAMES:
            out.append("actions.%s: 메타 도구 이름은 meta_tools 에" % n)
        if not isinstance(a, Mapping):
            out.append("actions.%s: 객체가 아님" % n)
            continue
        if "description" in a and not (isinstance(a["description"], str) and a["description"].strip()):
            out.append("actions.%s.description 은 비어 있지 않은 문자열" % n)
        if "schema" in a:
            out += check_schema("actions.%s" % n, a["schema"])
        out += ["actions.%s: 모르는 키 %s" % (n, k) for k in sorted(set(a) - {"description", "schema"})]
    meta = d.get("meta_tools", {})
    if not isinstance(meta, Mapping):
        out.append("meta_tools 는 객체")
        meta = {}
    for n, m in meta.items():
        if n not in META_TOOL_NAMES:
            out.append("meta_tools.%s: 메타 도구는 %s 뿐" % (n, META_TOOL_NAMES))
        elif not (isinstance(m, Mapping) and set(m) <= {"description"} and isinstance(m.get("description", ""), str)):
            out.append("meta_tools.%s: {description} 만" % n)
    if "pilot_guide" in d:
        out += check_guide("pilot_guide", d["pilot_guide"])
    if "progress_keys" in d:
        out += check_progress_keys("progress_keys", d["progress_keys"])
    if "observation_drop" in d:
        out += check_observation_drop("observation_drop", d["observation_drop"])
    return out


@dataclass
class PlaySpec:
    """외부 스펙 (검사 통과한 것). 빈 PlaySpec = 외부 스펙 없음."""
    actions: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    meta_tools: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    pilot_guide: Dict[str, Any] = field(default_factory=dict)
    progress_keys: Optional[List[str]] = None
    observation_drop: List[str] = field(default_factory=list)
    source: str = ""

    @classmethod
    def from_dict(cls, d: Mapping[str, Any], source: str = "") -> "PlaySpec":
        problems = validate_spec(d)
        if problems:
            raise SpecError("%s: %s" % (source or "스펙", "; ".join(problems)))
        return cls(actions={k: dict(v) for k, v in (d.get("actions") or {}).items()},
                   meta_tools={k: dict(v) for k, v in (d.get("meta_tools") or {}).items()},
                   pilot_guide=dict(d.get("pilot_guide") or {}),
                   progress_keys=list(d["progress_keys"]) if d.get("progress_keys") else None,
                   observation_drop=list(d.get("observation_drop") or []),
                   source=source)

    @classmethod
    def load(cls, path: Union[str, Path]) -> "PlaySpec":
        p = Path(path)
        return cls.from_dict(json.loads(p.read_text(encoding="utf-8")), source=str(p))


def as_spec(spec: Union[None, "PlaySpec", Mapping[str, Any], str, Path]) -> PlaySpec:
    """None / PlaySpec / dict / 파일 경로 → PlaySpec."""
    if spec is None:
        return PlaySpec()
    if isinstance(spec, PlaySpec):
        return spec
    if isinstance(spec, Mapping):
        return PlaySpec.from_dict(spec, source="dict")
    return PlaySpec.load(spec)


def effective_guide(description: Mapping[str, Any], spec: Optional[PlaySpec] = None) -> Dict[str, Any]:
    """게임 pilot_guide 와 외부 스펙 pilot_guide 를 키마다 합친다 (게임이 우선)."""
    out = dict((spec or PlaySpec()).pilot_guide)
    g = description.get("pilot_guide")
    if isinstance(g, Mapping):
        out.update({k: v for k, v in g.items() if v not in (None, "", [])})
    return out


def effective_progress_keys(description: Mapping[str, Any], spec: Optional[PlaySpec] = None) -> Optional[List[str]]:
    keys = description.get("progress_keys")
    if isinstance(keys, list) and keys:
        return [str(k) for k in keys]
    return (spec or PlaySpec()).progress_keys


def effective_observation_drop(description: Mapping[str, Any], spec: Optional[PlaySpec] = None) -> List[str]:
    """게임 선언 observation_drop 이 있으면 그것, 없으면 외부 스펙 것."""
    paths = description.get("observation_drop")
    if isinstance(paths, list) and paths:
        return [str(p) for p in paths]
    return list((spec or PlaySpec()).observation_drop)


__all__ = ["SPEC_FORMAT", "PlaySpec", "SpecError", "validate_spec", "as_spec", "effective_guide", "effective_progress_keys",
           "effective_observation_drop", "check_schema", "check_guide", "check_progress_keys", "check_observation_drop",
           "META_TOOL_NAMES"]
