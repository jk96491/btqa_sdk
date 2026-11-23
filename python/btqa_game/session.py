"""연결 1개에 대한 프로토콜 세션 — 게임 쪽 SDK.

  hello → hello_ok, describe → description,
  run_episode → (trigger ↔ trigger_response)* → episode_result, bye → 종료

게임 고유 로직은 `EpisodeHost` 에만 있다. Unity 판은 `unity/com.btqa.game` 의 `QaSession`.
"""
from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional

from btqa_game import protocol as P
from btqa_game.declaration import GameDecl
from btqa_game.host import EpisodeHost, EpisodeRequest, Recv, Send, TriggerChannel


class GameSession:
    def __init__(self, host: EpisodeHost) -> None:
        self.host = host
        self._decl: Optional[GameDecl] = None

    @property
    def decl(self) -> GameDecl:
        if self._decl is None:
            self._decl = self.host.declaration().validate()
        return self._decl

    # ------------------------------------------------------------------ 요청 검증
    def parse_request(self, msg: Mapping[str, Any]) -> EpisodeRequest:
        for key in ("episode_id", "class", "seed"):
            if key not in msg:
                raise P.ProtocolError(P.E_BAD_REQUEST, "run_episode 에 %r 필요" % key)
        cdecl = self.decl.class_decl(msg["class"])
        if cdecl is None:
            raise P.ProtocolError(P.E_UNKNOWN_CLASS, "알 수 없는 class %r" % (msg["class"],))
        scenario = msg.get("scenario") or "base"
        if not self.decl.has_scenario(scenario):
            raise P.ProtocolError(P.E_UNKNOWN_SCENARIO, "알 수 없는 scenario %r" % (scenario,))
        seed = msg["seed"]
        if isinstance(seed, bool) or not isinstance(seed, int):
            raise P.ProtocolError(P.E_BAD_REQUEST, "seed 는 정수여야 함")
        params = msg.get("params") or {}
        if not isinstance(params, dict):
            raise P.ProtocolError(P.E_BAD_REQUEST, "params 는 객체여야 함")
        return EpisodeRequest(
            episode_id=msg["episode_id"],
            class_decl=cdecl,
            scenario=str(scenario),
            seed=int(seed),
            params=cdecl.resolve(cdecl.defaults(), params),
            thresholds=self.parse_triggers(msg.get("triggers")),
            record_events=bool(msg.get("record_events")),
            raw=msg,
        )

    def parse_triggers(self, t: Any) -> List[float]:
        if not t:
            return []
        if not isinstance(t, dict):
            raise P.ProtocolError(P.E_BAD_REQUEST, "triggers 는 객체여야 함")
        key = self.decl.triggers.type
        vals = t.get(key) or []
        if not isinstance(vals, list) or any(isinstance(v, bool) or not isinstance(v, (int, float)) for v in vals):
            raise P.ProtocolError(P.E_BAD_REQUEST, "triggers.%s 는 숫자 배열이어야 함" % key)
        return [float(v) for v in vals]

    # ------------------------------------------------------------------ run_episode
    def run_episode(self, msg: Mapping[str, Any], send: Send, recv: Recv) -> Dict[str, Any]:
        req = self.parse_request(msg)
        result = self.host.run_episode(req, TriggerChannel(req, send, recv))
        return {"type": P.EPISODE_RESULT, "episode_id": req.episode_id, "result": result}

    def hello_ok(self) -> Dict[str, Any]:
        d = self.decl
        return {"type": P.HELLO_OK, "protocol_version": P.PROTOCOL_VERSION, "game": d.game, "game_version": d.game_version}

    # ------------------------------------------------------------------ 디스패치
    def handle(self, msg: Mapping[str, Any], send: Send, recv: Recv) -> bool:
        """메시지 1개 처리. 연결을 닫아야 하면 False."""
        kind = msg.get("type")
        try:
            if kind == P.HELLO:
                if P.major(msg.get("protocol_version", "")) != P.major(P.PROTOCOL_VERSION):
                    raise P.ProtocolError(P.E_PROTOCOL_VERSION, "지원 버전 %s, 요청 %r" % (P.PROTOCOL_VERSION, msg.get("protocol_version")))
                send(self.hello_ok())
            elif kind == P.DESCRIBE:
                send(self.decl.to_wire())
            elif kind == P.RUN_EPISODE:
                send(self.run_episode(msg, send, recv))
            elif kind == P.BYE:
                return False
            else:
                raise P.ProtocolError(P.E_UNKNOWN_TYPE, "알 수 없는 type %r" % (kind,))
        except P.ProtocolError as exc:
            send(P.error(exc.code, exc.message, episode_id=msg.get("episode_id")))
        except ConnectionError:
            return False
        except Exception as exc:  # 게임 내부 오류도 연결은 유지
            send(P.error(P.E_INTERNAL, "%s: %s" % (type(exc).__name__, exc), episode_id=msg.get("episode_id")))
        return True
