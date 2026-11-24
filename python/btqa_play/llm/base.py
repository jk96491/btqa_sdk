"""LLM 백엔드 추상화 — 각자 쓰는 LLM 을 끼워 넣는 확장 지점 (docs/PLAY_SDK.md 4장).

LlmPilot 은 이 인터페이스만 안다. 백엔드가 대화 기록을 들고 있고, 자기 API 형식으로 바꾼다.
도구 정의는 tools.build_tools() 의 중립 형식 {name, description, input_schema[, strict]}.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional

#: Turn.stop 값 (백엔드마다 다른 종료 이유를 여기로 맞춘다)
STOP_TOOL_USE = "tool_use"     # 도구를 불렀다
STOP_END = "end"               # 도구 없이 말만 하고 끝냄
STOP_MAX_TOKENS = "max_tokens"
STOP_REFUSAL = "refusal"
STOP_PAUSE = "pause"           # 서버가 잠깐 멈춤 → 그대로 다시 부르면 이어짐


@dataclass
class ToolCall:
    id: str
    name: str
    input: Dict[str, Any]
    error: Optional[str] = None    # 인자 JSON 파싱 실패 등 → 실행하지 않고 오류 결과로 돌려준다


@dataclass
class ToolResult:
    call_id: str
    content: str
    is_error: bool = False


@dataclass
class Turn:
    texts: List[str] = field(default_factory=list)
    calls: List[ToolCall] = field(default_factory=list)
    stop: str = STOP_END
    usage: Dict[str, int] = field(default_factory=dict)


class LlmBackend(ABC):
    """대화 1개 = 백엔드 인스턴스 1개."""

    #: 리포트에 남길 이름 (예: "anthropic:claude-opus-5-5", "openai:qwen2.5:14b@http://localhost:11434/v1")
    label: str = "llm"

    def __init__(self) -> None:
        self.usage: Dict[str, int] = {"input_tokens": 0, "output_tokens": 0}

    @abstractmethod
    def start(self, system: str, tools: List[Mapping[str, Any]], first_user: str) -> None:
        """대화 시작 (시스템 프롬프트·도구는 세션 내내 고정)."""

    @abstractmethod
    def complete(self) -> Turn:
        """지금까지의 대화로 모델을 한 번 부르고, 그 응답을 기록에 붙인 뒤 돌려준다."""

    @abstractmethod
    def add_tool_results(self, results: List[ToolResult]) -> None:
        """직전 Turn 의 도구 호출 결과를 (한 번에 모두) 붙인다."""

    @abstractmethod
    def add_user(self, text: str) -> None:
        """사용자 말 한 줄 (도구 없이 끝난 응답 뒤 상기 등)."""

    def _count(self, usage: Mapping[str, int]) -> None:
        for k, v in usage.items():
            self.usage[k] = self.usage.get(k, 0) + int(v or 0)
