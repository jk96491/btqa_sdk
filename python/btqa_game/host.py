"""게임이 구현하는 확장 지점 — 게임 쪽 SDK.

게임 개발자가 할 일은 `EpisodeHost` 하나를 구현하는 것이다:
  - `declaration()` : 정의서 (직업·파라미터·**블록 가중치**·시나리오·트리거)
  - `run_episode()` : 판 1개를 seed 로 결정론적으로 끝까지 실행. 트리거 경계마다 `triggers.fire(...)` 호출

요청 검증(class/scenario/seed/params 형식·범위), 트리거 왕복 검사, 오류 응답, 연결 유지는 SDK(`GameSession`)가 한다.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Mapping, Optional

from btqa_game import protocol as P
from btqa_game.declaration import ClassDecl, GameDecl

Send = Callable[[Dict[str, Any]], None]
Recv = Callable[[], Optional[Dict[str, Any]]]


@dataclass(frozen=True)
class EpisodeRequest:
    """검증을 마친 run_episode 요청."""

    episode_id: Any
    class_decl: ClassDecl
    scenario: str
    seed: int
    params: Dict[str, float]          # 정의서 default + 요청 params (검증 완료)
    thresholds: List[float]           # 요청한 트리거 경계 (없으면 빈 목록 = 트리거 없음)
    record_events: bool = False
    raw: Optional[Mapping[str, Any]] = None
    initial_state: Optional[Dict[str, float]] = None   # v1.4: 선언 전체(default + 요청, 검증 완료). None = 요청 없음 (1.3 과 같은 시작)

    @property
    def class_name(self) -> str:
        return self.class_decl.name


class TriggerChannel:
    """판 진행 중 트리거 왕복. 게임은 경계에서 `fire` 를 부르고 돌려받은 params 로 덮어쓴다 (응답까지 게임 시간 정지)."""

    def __init__(self, request: EpisodeRequest, send: Send, recv: Recv) -> None:
        self.request = request
        self._send = send
        self._recv = recv
        self.fired = 0

    def fire(self, index: int, state: Mapping[str, float], totals: Mapping[str, float]) -> Dict[str, float]:
        """trigger 를 보내고 trigger_response 를 기다린다. 반환 = 현재 params 위에 덮어쓸 값 (검증 완료)."""
        eid = self.request.episode_id
        self._send({"type": P.TRIGGER, "episode_id": eid, "index": int(index),
                    "state": dict(state), "totals": dict(totals)})
        reply = self._recv()
        if reply is None:
            raise ConnectionError("trigger 응답 대기 중 연결 종료")
        if reply.get("type") != P.TRIGGER_RESPONSE or reply.get("episode_id") != eid:
            raise P.ProtocolError(P.E_BAD_REQUEST, "trigger_response(episode_id=%r) 를 기대했으나 %r" % (eid, reply.get("type")))
        if reply.get("index") != index:
            raise P.ProtocolError(P.E_BAD_REQUEST, "trigger_response index %r != %r" % (reply.get("index"), index))
        params = reply.get("params") or {}
        if not isinstance(params, dict):
            raise P.ProtocolError(P.E_BAD_REQUEST, "trigger_response.params 는 객체여야 함")
        self.fired += 1
        return self.request.class_decl.resolve({}, params)


class EpisodeHost(ABC):
    @abstractmethod
    def declaration(self) -> GameDecl:
        """정의서. SDK 가 validate() 후 describe 응답으로 보낸다."""

    @abstractmethod
    def run_episode(self, request: EpisodeRequest, triggers: TriggerChannel) -> Dict[str, Any]:
        """판 1개 → episode_result.result (docs/PROTOCOL.md 4.5)."""
