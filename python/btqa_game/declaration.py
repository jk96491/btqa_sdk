"""정의서 (describe 응답) 선언 모델 — 게임 쪽 SDK.

SDK 의 핵심 규칙 (사용자 결정 2026-11-23):

    학습 대상 = 전략 선택 노드의 블록 가중치 w (S = N × w) 뿐이다.

- 게임은 BT 의 블록과 그 가중치 파라미터를 `blocks` 로 선언한다.
- `optimize` 는 게임이 고르지 않는다. 블록 가중치면 true, 아니면 false 로 SDK 가 정한다.
- 가중치가 아닌 BT 값(물약 체력, 자원 비축, 최소 유지 시간 등)과 페르소나 값(θ, φ)은
  기획(default)·페르소나가 정하고 학습하지 않는다.

`GameDecl.validate()` 가 이 규칙을 검사하고, `to_wire()` 가 describe 응답을 만든다.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from btqa_game import protocol as P

GROUP_BT = "bt"            # BT 값 (가중치 또는 기획이 정하는 설정값)
GROUP_PERSONA = "persona"  # 페르소나가 정하는 필요도 설정값 (θ, φ 등)
PARAM_GROUPS = (GROUP_BT, GROUP_PERSONA)
WEIGHT_UNIT = "weight"


class DeclarationError(ValueError):
    def __init__(self, problems: Sequence[str]) -> None:
        self.problems = list(problems)
        super().__init__("정의서 규칙 위반: " + "; ".join(self.problems))


@dataclass(frozen=True)
class ParamDecl:
    name: str
    low: float
    high: float
    default: float                 # 개발팀 손 튜닝 값 (기준선)
    group: str = GROUP_BT
    unit: str = ""
    description: str = ""
    optimize: Optional[bool] = None   # 정의서에 적힌 값 (검사용). wire 의 optimize 는 SDK 가 정한다

    def contains(self, value: float, tol: float = 1e-9) -> bool:
        return self.low - tol <= value <= self.high + tol


@dataclass(frozen=True)
class BlockDecl:
    """전략 선택 노드의 블록 1개. 점수 S = N(상황) × weight."""

    name: str
    weight: str                    # 가중치 파라미터 이름
    description: str = ""

    def to_wire(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {"name": self.name, "weight": self.weight}
        if self.description:
            d["description"] = self.description
        return d


@dataclass(frozen=True)
class ClassDecl:
    name: str
    params: Tuple[ParamDecl, ...]
    blocks: Tuple[BlockDecl, ...]
    display_name: str = ""
    ranged: bool = False
    spec_version: str = "0.1"

    @property
    def weight_names(self) -> List[str]:
        return [b.weight for b in self.blocks]

    def param(self, name: str) -> Optional[ParamDecl]:
        for p in self.params:
            if p.name == name:
                return p
        return None

    def problems(self) -> List[str]:
        out: List[str] = []
        names = [p.name for p in self.params]
        if len(set(names)) != len(names):
            out.append("%s: 파라미터 이름 중복" % self.name)
        for p in self.params:
            if p.group not in PARAM_GROUPS:
                out.append("%s.%s: group 은 %s" % (self.name, p.name, "|".join(PARAM_GROUPS)))
            if not (p.low < p.high and p.low <= p.default <= p.high):
                out.append("%s.%s: range/default 오류" % (self.name, p.name))
        if not self.blocks:
            out.append("%s: blocks 가 비어 있음 (학습할 가중치가 없음)" % self.name)
        bnames = [b.name for b in self.blocks]
        if len(set(bnames)) != len(bnames):
            out.append("%s: 블록 이름 중복" % self.name)
        weights = self.weight_names
        if len(set(weights)) != len(weights):
            out.append("%s: 블록 두 개가 같은 가중치를 씀" % self.name)
        for b in self.blocks:
            p = self.param(b.weight)
            if p is None:
                out.append("%s: 블록 %s 의 가중치 %r 가 params 에 없음" % (self.name, b.name, b.weight))
                continue
            if p.group != GROUP_BT:
                out.append("%s.%s: 가중치는 group bt 여야 함" % (self.name, p.name))
            if p.low < 0:
                out.append("%s.%s: 가중치 범위는 0 이상" % (self.name, p.name))
            if p.unit not in ("", WEIGHT_UNIT):
                out.append("%s.%s: 가중치 unit 은 %r" % (self.name, p.name, WEIGHT_UNIT))
            if p.optimize is False:
                out.append("%s.%s: 블록 가중치를 optimize=false 로 둘 수 없음" % (self.name, p.name))
        for p in self.params:
            if p.name not in weights and p.optimize:
                out.append("%s.%s: 가중치가 아닌 파라미터는 학습 대상이 될 수 없음 (optimize=true 금지)" % (self.name, p.name))
        return out

    def to_wire(self) -> Dict[str, Any]:
        weights = set(self.weight_names)
        params = []
        for p in self.params:
            is_w = p.name in weights
            params.append({
                "name": p.name,
                "range": [p.low, p.high],
                "default": p.default,
                "optimize": is_w,
                "group": p.group,
                "unit": WEIGHT_UNIT if is_w else p.unit,
                "description": p.description,
            })
        return {
            "name": self.name,
            "display_name": self.display_name or self.name,
            "ranged": self.ranged,
            "spec_version": self.spec_version,
            "params": params,
            "blocks": [b.to_wire() for b in self.blocks],
        }

    # ------------------------------------------------------------------ 판 파라미터
    def defaults(self) -> Dict[str, float]:
        return {p.name: p.default for p in self.params}

    def resolve(self, base: Mapping[str, float], updates: Mapping[str, Any]) -> Dict[str, float]:
        """base 위에 updates 덮어쓰기. 모르는 이름·범위 밖·숫자 아님 → ProtocolError."""
        out = dict(base)
        for name, value in updates.items():
            p = self.param(name)
            if p is None:
                raise P.ProtocolError(P.E_UNKNOWN_PARAM, "class %r 에 파라미터 %r 없음" % (self.name, name))
            if isinstance(value, bool):
                raise P.ProtocolError(P.E_BAD_REQUEST, "파라미터 %r 값이 숫자가 아님: %r" % (name, value))
            try:
                v = float(value)
            except (TypeError, ValueError):
                raise P.ProtocolError(P.E_BAD_REQUEST, "파라미터 %r 값이 숫자가 아님: %r" % (name, value))
            if not p.contains(v):
                raise P.ProtocolError(P.E_PARAM_OUT_OF_RANGE, "%s=%g 가 범위 [%g, %g] 밖" % (name, v, p.low, p.high))
            out[name] = v
        return out


@dataclass(frozen=True)
class TriggerDecl:
    type: str                                  # run_episode.triggers 의 키 (예: boss_hp_below)
    supported: Tuple[float, ...] = ()          # 기본 경계
    state_features: Tuple[str, ...] = ()       # 정책 입력 순서 (모두 0~1)
    totals: Tuple[str, ...] = ()               # trigger.totals 키 (v1.1)

    def to_wire(self) -> Dict[str, Any]:
        return {
            "type": self.type,
            "supported": list(self.supported),
            "state_features": list(self.state_features),
            "totals": list(self.totals),
        }


@dataclass(frozen=True)
class InitialStateDecl:
    """v1.4 `run_episode.initial_state` 로 받을 수 있는 판 시작 상태 1개.

    이름은 `triggers.state_features` 중 하나 (0~1 비율, 예: hp_ratio, potions_ratio) — 트리거 state 와 같은 뜻.
    생략한 키 = default (보통 1.0 = 가득). `initial_state` 를 아예 안 보낸 판은 1.3 과 같은 결과여야 한다.
    """

    name: str
    low: float = 0.0
    high: float = 1.0
    default: float = 1.0
    description: str = ""

    def contains(self, value: float, tol: float = 1e-9) -> bool:
        return self.low - tol <= value <= self.high + tol

    def to_wire(self) -> Dict[str, Any]:
        return {"name": self.name, "range": [self.low, self.high], "default": self.default, "description": self.description}


@dataclass(frozen=True)
class ScenarioDecl:
    name: str
    description: str = ""


@dataclass(frozen=True)
class GameDecl:
    game: str
    game_version: str
    time_limit: float
    max_deaths: int
    classes: Tuple[ClassDecl, ...]
    scenarios: Tuple[ScenarioDecl, ...]
    triggers: TriggerDecl
    decision_interval: float = 0.0
    result_fields: Tuple[str, ...] = ()
    extra: Dict[str, Any] = field(default_factory=dict)   # 게임 고유 추가 필드 (학습 쪽은 모르는 필드를 무시)
    initial_state: Tuple[InitialStateDecl, ...] = ()      # v1.4: 받을 수 있는 판 시작 상태 (없으면 initial_state 요청 거부)

    def class_decl(self, name: Any) -> Optional[ClassDecl]:
        for c in self.classes:
            if c.name == name:
                return c
        return None

    def has_scenario(self, name: Any) -> bool:
        return any(s.name == name for s in self.scenarios)

    def problems(self) -> List[str]:
        out: List[str] = []
        if not self.classes:
            out.append("classes 가 비어 있음")
        for c in self.classes:
            out += c.problems()
        if not self.has_scenario("base"):
            out.append("scenario 'base' 필수")
        if not self.triggers.type:
            out.append("triggers.type 필요")
        if not self.triggers.state_features:
            out.append("triggers.state_features 비어 있음")
        names = [s.name for s in self.initial_state]
        if len(set(names)) != len(names):
            out.append("initial_state 이름 중복")
        for s in self.initial_state:
            if s.name not in self.triggers.state_features:
                out.append("initial_state.%s: triggers.state_features 에 없는 이름 (같은 뜻의 0~1 비율이어야 함)" % s.name)
            if not (0.0 <= s.low < s.high <= 1.0 and s.low <= s.default <= s.high):
                out.append("initial_state.%s: range 는 [0,1] 안, low < high, default 는 range 안" % s.name)
        return out

    def resolve_initial_state(self, value: Any) -> Optional[Dict[str, float]]:
        """run_episode.initial_state 검증 → 선언 전체(default + 요청). 없거나 빈 객체면 None (= 1.3 과 같은 시작)."""
        if value is None or value == {}:
            return None
        if not isinstance(value, dict):
            raise P.ProtocolError(P.E_BAD_REQUEST, "initial_state 는 객체여야 함")
        decls = {s.name: s for s in self.initial_state}
        out = {s.name: s.default for s in self.initial_state}
        for name, v in value.items():
            s = decls.get(name)
            if s is None:
                raise P.ProtocolError(P.E_UNKNOWN_PARAM, "initial_state 에 %r 없음 (선언: %s)" % (name, sorted(decls) or "없음"))
            if isinstance(v, bool) or not isinstance(v, (int, float)):
                raise P.ProtocolError(P.E_BAD_REQUEST, "initial_state.%s 값이 숫자가 아님: %r" % (name, v))
            if not s.contains(float(v)):
                raise P.ProtocolError(P.E_PARAM_OUT_OF_RANGE, "initial_state.%s=%g 가 범위 [%g, %g] 밖" % (name, v, s.low, s.high))
            out[name] = float(v)
        return out

    def validate(self) -> "GameDecl":
        problems = self.problems()
        if problems:
            raise DeclarationError(problems)
        return self

    def to_wire(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {
            "type": P.DESCRIPTION,
            "protocol_version": P.PROTOCOL_VERSION,
            "game": self.game,
            "game_version": self.game_version,
            "time_limit": self.time_limit,
            "max_deaths": self.max_deaths,
            "decision_interval": self.decision_interval,
            "classes": [c.to_wire() for c in self.classes],
            "scenarios": [{"name": s.name, "description": s.description} for s in self.scenarios],
            "triggers": self.triggers.to_wire(),
        }
        if self.result_fields:
            d["result_fields"] = list(self.result_fields)
        if self.initial_state:
            d["initial_state"] = [s.to_wire() for s in self.initial_state]
        for k, v in self.extra.items():
            d.setdefault(k, v)
        return d
