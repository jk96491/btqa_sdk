"""게임 ↔ 학습 플랫폼 프로토콜 상수·인코딩 (게임 쪽 SDK). 명세: docs/PROTOCOL.md

전송: TCP, UTF-8 JSON 한 줄 = 메시지 1개 (JSON Lines, '\\n' 구분). 모든 메시지는 {"type": ...} 를 가진다.
"""
from __future__ import annotations

import json
from typing import Any, Dict

#: 1.1 trigger.totals / 1.2 result.stalled / 1.3 classes[].blocks (학습 대상 = 블록 가중치)
PROTOCOL_VERSION = "1.3"

# 학습 → 게임
HELLO = "hello"
DESCRIBE = "describe"
RUN_EPISODE = "run_episode"
TRIGGER_RESPONSE = "trigger_response"
BYE = "bye"

# 게임 → 학습
HELLO_OK = "hello_ok"
DESCRIPTION = "description"
TRIGGER = "trigger"
EPISODE_RESULT = "episode_result"
ERROR = "error"

# 오류 코드
E_BAD_REQUEST = "bad_request"
E_UNKNOWN_TYPE = "unknown_type"
E_PROTOCOL_VERSION = "protocol_version"
E_UNKNOWN_CLASS = "unknown_class"
E_UNKNOWN_SCENARIO = "unknown_scenario"
E_UNKNOWN_PARAM = "unknown_param"
E_PARAM_OUT_OF_RANGE = "param_out_of_range"
E_INTERNAL = "internal"


class ProtocolError(Exception):
    """error 응답으로 바뀌는 예외. 연결은 유지된다."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message


def encode(msg: Dict[str, Any]) -> bytes:
    return (json.dumps(msg, ensure_ascii=False, separators=(",", ":")) + "\n").encode("utf-8")


def decode(line: bytes) -> Dict[str, Any]:
    try:
        msg = json.loads(line.decode("utf-8"))
    except (UnicodeDecodeError, ValueError) as exc:
        raise ProtocolError(E_BAD_REQUEST, "JSON 파싱 실패: %s" % exc)
    if not isinstance(msg, dict) or "type" not in msg:
        raise ProtocolError(E_BAD_REQUEST, "메시지는 'type' 을 가진 JSON 객체여야 함")
    return msg


def error(code: str, message: str, **extra: Any) -> Dict[str, Any]:
    msg = {"type": ERROR, "code": code, "message": message}
    msg.update({k: v for k, v in extra.items() if v is not None})
    return msg


def major(version: str) -> str:
    return str(version).split(".")[0]
