"""Claude API 백엔드 (anthropic SDK, Python ≥ 3.10 별도 venv). 키 = 환경변수 ANTHROPIC_API_KEY (SDK 가 읽음).

- thinking 상시(Opus 5.5), 깊이는 output_config.effort, tool_choice auto + strict 도구
- 시스템 블록 캐시 지점 + 상위 cache_control(대화 꼬리 자동 캐시)
"""
from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional

from btqa_play.llm.base import (STOP_END, STOP_MAX_TOKENS, STOP_PAUSE, STOP_REFUSAL, STOP_TOOL_USE,
                                 LlmBackend, ToolCall, ToolResult, Turn)

DEFAULT_MODEL = "claude-opus-5-5"
_STOP = {"tool_use": STOP_TOOL_USE, "end_turn": STOP_END, "stop_sequence": STOP_END,
         "max_tokens": STOP_MAX_TOKENS, "refusal": STOP_REFUSAL, "pause_turn": STOP_PAUSE}
_USAGE_KEYS = ("input_tokens", "output_tokens", "cache_read_input_tokens", "cache_creation_input_tokens")


class AnthropicBackend(LlmBackend):
    def __init__(self, model: str = DEFAULT_MODEL, effort: str = "low", max_tokens: int = 8000,
                 api_client: Any = None, api_key_env: Optional[str] = None) -> None:
        super().__init__()
        self.model, self.effort, self.max_tokens = model, effort, max_tokens
        self.label = "anthropic:" + model
        self._api = api_client          # 테스트에서 가짜 주입 (messages.create 만 쓴다)
        self._api_key_env = api_key_env
        self.usage = {k: 0 for k in _USAGE_KEYS}
        self.system: List[Dict[str, Any]] = []
        self.tools: List[Dict[str, Any]] = []
        self.messages: List[Dict[str, Any]] = []

    def _client(self) -> Any:
        if self._api is None:
            import os
            import anthropic   # Python ≥ 3.10 전용 venv
            key = os.environ.get(self._api_key_env) if self._api_key_env else None
            self._api = anthropic.Anthropic(api_key=key) if key else anthropic.Anthropic()
        return self._api

    def start(self, system: str, tools: List[Mapping[str, Any]], first_user: str) -> None:
        self.system = [{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}]
        self.tools = [dict(t) for t in tools]
        self.messages = [{"role": "user", "content": first_user}]

    def complete(self) -> Turn:
        resp = self._client().messages.create(
            model=self.model, max_tokens=self.max_tokens, system=self.system, tools=self.tools, messages=self.messages,
            output_config={"effort": self.effort}, cache_control={"type": "ephemeral"},
        )
        u = {k: int(getattr(resp.usage, k, 0) or 0) for k in _USAGE_KEYS}
        self._count(u)
        self.messages.append({"role": "assistant", "content": resp.content})   # thinking 블록 포함 그대로
        texts = [b.text for b in resp.content if getattr(b, "type", "") == "text" and b.text.strip()]
        calls = [ToolCall(b.id, b.name, dict(b.input)) for b in resp.content if getattr(b, "type", "") == "tool_use"]
        return Turn(texts=texts, calls=calls, stop=_STOP.get(str(resp.stop_reason), STOP_END), usage=u)

    def add_tool_results(self, results: List[ToolResult]) -> None:
        self.messages.append({"role": "user", "content": [
            {"type": "tool_result", "tool_use_id": r.call_id, "content": r.content, "is_error": r.is_error} for r in results]})

    def add_user(self, text: str) -> None:
        self.messages.append({"role": "user", "content": text})
