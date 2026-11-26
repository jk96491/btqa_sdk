"""플레이 모드 프로토콜(docs/PLAY_PROTOCOL.md v0.3) 클라이언트 — BT-QA Play SDK.

- PlayClient.act() 는 action_result(또는 error)까지 읽는다. 그 사이 오는 전투 메시지(combat_started·trigger·
  combat_ended)를 모으고, trigger 는 응답기(TriggerResponder)로 **즉시** 답한다 — LLM 을 거치지 않는다.
- 전투 중 error 를 받으면 클라이언트 버그로 보고 PlayError 를 올린다 (게임은 같은 trigger 를 계속 기다리므로
  다시 보내지 않고 세션을 끝낸다. PLAY_PROTOCOL 4.7).
- 표준 라이브러리만 쓴다 (Python 3.8 호환).
"""
from __future__ import annotations

import json
import socket
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

PROTOCOL = "play"
PROTOCOL_VERSION = "0.3"
#: hello 거절 시 차례로 다시 시도. 0.6.2~0.7.0 은 "0.1" 만 받는다 (내용은 0.2 와 같고 미반영 항목만 다름).
#: 0.3 = 0.2 + pilot_note (4.10). 0.3 미만 게임에는 메모를 보내지 않고 클라이언트 쪽에만 남긴다.
COMPATIBLE_VERSIONS = ("0.2", "0.1")
NOTE_KINDS = ("thought", "plan", "finding")
#: 로드아웃 boss 불일치 코드 (0.6.2 = unknown_boss, v0.2 = boss_mismatch)
BOSS_MISMATCH_CODES = ("boss_mismatch", "unknown_boss")

#: trigger 메시지 → 응답 params(dict, 부분 덮어쓰기) 또는 None(바꾸지 않음)
TriggerResponder = Callable[[Mapping[str, Any]], Optional[Dict[str, float]]]


class PlayError(RuntimeError):
    """게임이 error 를 보냈거나 프로토콜이 어긋남."""

    def __init__(self, code: str, message: str = "", msg: Optional[Mapping[str, Any]] = None) -> None:
        super().__init__("%s: %s" % (code, message))
        self.code = code
        self.message = message
        self.msg = dict(msg or {})


def parse_endpoint(ep: str) -> Tuple[str, int]:
    host, port = ep.rsplit(":", 1)
    return host, int(port)


@dataclass
class Combat:
    """act 하나 안에서 일어난 전투 1회."""
    started: Dict[str, Any]
    triggers: List[Dict[str, Any]] = field(default_factory=list)      # 받은 trigger
    replies: List[Optional[Dict[str, float]]] = field(default_factory=list)  # 보낸 params
    ended: Optional[Dict[str, Any]] = None

    @property
    def encounter_id(self) -> str:
        return str(self.started.get("encounter_id"))

    @property
    def result(self) -> Dict[str, Any]:
        return dict((self.ended or {}).get("result") or {})

    @property
    def cleared(self) -> bool:
        return bool(self.result.get("cleared"))

    def summary(self) -> Dict[str, Any]:
        """LLM·리포트에 보여 줄 요약 (전투 세부는 BT 몫이라 결과만)."""
        r = self.result
        af = (self.ended or {}).get("aftermath") or {}
        return {
            "encounter_id": self.encounter_id,
            "enemy": self.started.get("enemy"),
            "scenario": self.started.get("scenario"),
            "seed": self.started.get("seed"),
            "loadout_matched": (self.started.get("loadout") or {}).get("matched"),
            "cleared": r.get("cleared"),
            "elapsed": r.get("elapsed"),
            "deaths": r.get("deaths"),
            "stalled": r.get("stalled"),
            "triggers_answered": len(self.replies),
            "loot": af.get("loot"),
            "xp": af.get("xp"),
            "respawned_at": af.get("respawned_at"),
            # v0.3: 전투 물약 = 인벤토리 (퀵슬롯 등록분 min(보유, cap), 사용분 차감). v0.2 이하 게임은 None
            "potions": self.started.get("potions"),
            "potions_used": af.get("potions_used"),
            "out_of_training_dist": _short_on_potions(self.started.get("potions")),
        }


def _short_on_potions(p: Any) -> Optional[bool]:
    """물약을 cap 보다 적게 가져간 전투 = 학습 판(항상 cap 개 시작) 분포 밖."""
    if not isinstance(p, Mapping) or p.get("carried") is None or p.get("cap") is None:
        return None
    return int(p["carried"]) < int(p["cap"])


@dataclass
class ActResult:
    """act 응답. msg = action_result 원문, combats = 그 사이 전투들."""
    msg: Dict[str, Any]
    combats: List[Combat] = field(default_factory=list)

    @property
    def status(self) -> Optional[str]:
        return self.msg.get("status")

    @property
    def reason(self) -> Optional[str]:
        return self.msg.get("reason")

    @property
    def observation(self) -> Dict[str, Any]:
        return dict(self.msg.get("observation") or {})

    @property
    def is_error(self) -> bool:
        return self.msg.get("type") == "error"

    def get(self, key: str, default: Any = None) -> Any:
        return self.msg.get(key, default)


class PlayClient:
    """연결 1개 = 세션 1개 (PLAY_PROTOCOL 2장)."""

    def __init__(self, host: str, port: int, timeout: float = 300.0, client_name: str = "btqa_play") -> None:
        self.host, self.port = host, int(port)
        self.client_name = client_name
        self._sock = socket.create_connection((host, self.port), timeout=timeout)
        self._f = self._sock.makefile("rwb")
        self.seq = 0
        self.hello_msg: Dict[str, Any] = {}
        self.description: Dict[str, Any] = {}
        self.protocol_version: Optional[str] = None
        self.notes: List[str] = []                    # 호환 시도 등 기록
        self.pilot_notes: List[Dict[str, Any]] = []   # 플레이어(LLM) 생각 메모 — 게임이 표시를 못 해도 여기는 남는다

    @classmethod
    def connect(cls, endpoint: str, **kw: Any) -> "PlayClient":
        h, p = parse_endpoint(endpoint)
        return cls(h, p, **kw)

    # ------------------------------------------------------------------ 전송
    def send(self, msg: Mapping[str, Any]) -> None:
        self._f.write((json.dumps(msg, ensure_ascii=False) + "\n").encode("utf-8"))
        self._f.flush()

    def send_raw(self, line: bytes) -> None:
        self._f.write(line)
        self._f.flush()

    def recv(self) -> Dict[str, Any]:
        line = self._f.readline()
        if not line:
            raise PlayError("disconnected", "게임이 연결을 끊음")
        return json.loads(line.decode("utf-8"))

    def request(self, msg: Mapping[str, Any]) -> Dict[str, Any]:
        """요청 1개 → 응답 1개 (error 도 그대로 반환)."""
        self.send(msg)
        return self.recv()

    def call(self, msg: Mapping[str, Any], expect: str) -> Dict[str, Any]:
        """request + 응답 type 확인. error 면 PlayError."""
        r = self.request(msg)
        if r.get("type") == "error":
            raise PlayError(str(r.get("code")), str(r.get("message", "")), r)
        if r.get("type") != expect:
            raise PlayError("unexpected", "%s 를 기대했는데 %s" % (expect, r.get("type")), r)
        return r

    # ------------------------------------------------------------------ 세션
    def hello(self, versions: Optional[Sequence[str]] = None) -> Dict[str, Any]:
        tried = []
        for v in list(versions or (PROTOCOL_VERSION,) + COMPATIBLE_VERSIONS):
            r = self.request({"type": "hello", "protocol": PROTOCOL, "protocol_version": v, "client": self.client_name})
            tried.append(v)
            if r.get("type") == "hello_ok":
                self.hello_msg = r
                self.protocol_version = v
                if v != PROTOCOL_VERSION:
                    self.notes.append("게임이 play %s 를 거절해 %s 로 접속 (%s %s)"
                                      % (PROTOCOL_VERSION, v, r.get("game"), r.get("game_version")))
                return r
            if r.get("code") != "protocol_version":
                raise PlayError(str(r.get("code")), str(r.get("message", "")), r)
        raise PlayError("protocol_version", "지원 버전 없음 (시도 %s)" % tried)

    def describe(self) -> Dict[str, Any]:
        self.description = self.call({"type": "play_describe"}, "play_description")
        return self.description

    def set_loadout(self, entries: Sequence[Mapping[str, Any]], persona: Optional[str] = None) -> Dict[str, Any]:
        msg: Dict[str, Any] = {"type": "set_loadout", "entries": [dict(e) for e in entries]}
        if persona:
            msg["persona"] = persona
        return self.call(msg, "loadout_ok")

    def reset(self, seed: Optional[int] = None, cls: Optional[str] = None,
              options: Optional[Mapping[str, Any]] = None) -> Dict[str, Any]:
        """options = 게임이 play_description.reset_options 로 선언한 시작 조건만 (4.4.1, 선택). 선언하지 않은 키는 보내기 전에 거부."""
        msg: Dict[str, Any] = {"type": "reset"}
        if seed is not None:
            msg["seed"] = int(seed)
        if cls is not None:
            msg["class"] = cls
        if options:
            unknown = sorted(set(options) - set(self.reset_option_names))
            if unknown:
                raise PlayError("unknown_reset_option", "게임이 선언하지 않은 reset 옵션 %s (선언: %s)"
                                % (unknown, sorted(self.reset_option_names)))
            msg["options"] = dict(options)
        return self.call(msg, "observation")

    @property
    def reset_option_names(self) -> List[str]:
        """play_description.reset_options 선언 이름 (4.4.1). 선언이 없으면 빈 목록."""
        ro = self.description.get("reset_options") or []
        return [str(o.get("name")) for o in ro if isinstance(o, Mapping) and o.get("name")]

    @property
    def supports_snapshot(self) -> bool:
        """게임이 snapshot/restore(4.11, 선택)를 선언했는가."""
        return bool((self.description.get("capabilities") or {}).get("snapshot"))

    def snapshot(self) -> str:
        """지금 게임 상태를 저장하고 id 를 받는다 (4.11). 같은 연결 안에서만 유효."""
        return str(self.call({"type": "snapshot"}, "snapshot_ok")["id"])

    def restore(self, snapshot_id: str) -> Dict[str, Any]:
        """저장한 상태로 되돌리고 그때의 관찰을 받는다 (4.11)."""
        return self.call({"type": "restore", "id": snapshot_id}, "observation")

    def observe(self, mode: str = "semantic", **kw: Any) -> Dict[str, Any]:
        return self.call(dict({"type": "observe", "mode": mode}, **kw), "observation")

    def get_log(self, since_seq: int = 0) -> Dict[str, Any]:
        return self.call({"type": "get_log", "since_seq": since_seq}, "play_log")

    # ------------------------------------------------------------------ 행동 (+ 전투)
    def act(self, action: str, args: Optional[Mapping[str, Any]] = None,
            responder: Optional[TriggerResponder] = None) -> ActResult:
        """act 를 보내고 action_result 또는 error 까지 읽는다.

        전투 메시지는 ActResult.combats 에 모은다. trigger 는 responder(msg) 로 바로 답한다 (None = 바꾸지 않음).
        responder 에 on_combat_started(msg) 가 있으면 조우 시작 때 불러 준다 (CombatResponder).
        전투 중 error(잘못된 trigger_response 등) → PlayError (PLAY_PROTOCOL 4.7 클라이언트 규칙).
        """
        self.seq += 1
        self.send({"type": "act", "seq": self.seq, "action": action, "args": dict(args or {})})
        combats: List[Combat] = []
        cur: Optional[Combat] = None
        while True:
            m = self.recv()
            t = m.get("type")
            if t == "combat_started":
                cur = Combat(started=m)
                combats.append(cur)
                hook = getattr(responder, "on_combat_started", None)
                if hook is not None:
                    hook(m)   # 응답기가 이 조우의 (class, scenario) 정책을 고른다
            elif t == "trigger":
                if cur is None:
                    raise PlayError("unexpected", "combat_started 없이 trigger", m)
                params = responder(m) if responder is not None else None
                cur.triggers.append(m)
                cur.replies.append(params)
                self.send({"type": "trigger_response", "encounter_id": m["encounter_id"], "index": m["index"],
                           "params": params})
            elif t == "combat_ended":
                if cur is not None:
                    cur.ended = m
                cur = None
            elif t == "error" and cur is not None:
                raise PlayError(str(m.get("code")), "전투 중 error: %s" % m.get("message", ""), m)
            else:
                return ActResult(msg=m, combats=combats)

    # ------------------------------------------------------------------ 플레이어 메모 (v0.3, 4.10)
    @property
    def supports_pilot_notes(self) -> bool:
        return bool(self.description.get("pilot_notes")) and self.protocol_version not in (None, "0.1", "0.2")

    def note(self, text: str, kind: str = "thought") -> bool:
        """플레이어 생각을 게임 화면·로그에 남긴다. 게임 상태·시간은 바뀌지 않는다. 표시했으면 True."""
        if kind not in NOTE_KINDS:
            raise ValueError("kind 는 %s 중 하나" % (NOTE_KINDS,))
        text = " ".join(str(text).split())
        limit = int((self.description.get("pilot_notes") or {}).get("max_chars", 200))
        if len(text) > limit:
            text = text[: limit - 1] + "…"
        rec: Dict[str, Any] = {"kind": kind, "text": text, "shown": False}
        self.pilot_notes.append(rec)
        if not text or not self.supports_pilot_notes:
            return False
        self.seq += 1
        self.call({"type": "pilot_note", "seq": self.seq, "kind": kind, "text": text}, "pilot_note_ok")
        rec["shown"] = True
        return True

    def close(self) -> None:
        try:
            self.send({"type": "bye"})
        except OSError:
            pass
        try:
            self._sock.close()
        except OSError:
            pass

    def __enter__(self) -> "PlayClient":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()
