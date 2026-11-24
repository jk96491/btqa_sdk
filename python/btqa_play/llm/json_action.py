"""JSON 행동 백엔드 — 서버의 도구 호출(function calling) 대신 **JSON 한 개를 스키마로 강제**해서 받는다.

작은 로컬 모델(0.5B~3B)은 도구 호출 형식을 자주 깬다. 이 백엔드는:
- 도구 목록·출력 형식을 system 메시지에 글로 넣고 (actions.action_system_prompt — 증류 데이터와 같은 글자)
- 출력을 response_format json_schema (actions.action_schema) 로 강제한다 → Ollama·llama.cpp 서버가 문법으로 막는다
  (서버가 json_schema 를 모르면 schema=False 로 끄고 파싱만: 형식이 깨지면 ToolCall.error)
- OpenAI 호환 /v1/chat/completions 를 그대로 쓴다 (표준 라이브러리만, Python 3.8)

LlmPilot(stateless=True) 와 같이 쓰는 것이 기본: 단계마다 start() 로 새 대화 = [system, user].
"""
from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional

from btqa_play.actions import action_schema, action_system_prompt, parse_action
from btqa_play.llm.base import STOP_END, STOP_TOOL_USE, ToolCall, ToolResult, Turn
from btqa_play.llm.openai_compat import DEFAULT_BASE_URL, OpenAICompatBackend, Post


class JsonActionBackend(OpenAICompatBackend):
    def __init__(self, model: str, base_url: str = DEFAULT_BASE_URL, api_key_env: Optional[str] = None,
                 max_tokens: int = 400, temperature: Optional[float] = 0.0, timeout: float = 600.0,
                 extra_body: Optional[Mapping[str, Any]] = None, post: Optional[Post] = None, schema: bool = True) -> None:
        super().__init__(model, base_url=base_url, api_key_env=api_key_env, max_tokens=max_tokens, temperature=temperature,
                         timeout=timeout, extra_body=extra_body, post=post)
        self.label = "openai-json:%s@%s" % (model, self.base_url)
        self.use_schema = schema
        self.tool_names: List[str] = []
        self._schema: Dict[str, Any] = {}
        self.system_text = ""

    def start(self, system: str, tools: List[Mapping[str, Any]], first_user: str) -> None:
        self.tool_names = [t["name"] for t in tools]
        self._schema = action_schema(tools)
        self.system_text = action_system_prompt(system, tools)
        self.tools = []   # 서버 도구 호출은 쓰지 않는다
        self.messages = [{"role": "system", "content": self.system_text}, {"role": "user", "content": first_user}]

    def complete(self) -> Turn:
        body: Dict[str, Any] = dict(self.extra_body, model=self.model, messages=self.messages, max_tokens=self.max_tokens)
        if self.temperature is not None:
            body["temperature"] = self.temperature
        if self.use_schema:
            body["response_format"] = {"type": "json_schema",
                                       "json_schema": {"name": "action", "schema": self._schema, "strict": True}}
        resp = self._post(self.base_url + "/chat/completions", self._headers(), body, self.timeout)
        choice = (resp.get("choices") or [{}])[0]
        text = ((choice.get("message") or {}).get("content") or "").strip()
        u = resp.get("usage") or {}
        usage = {"input_tokens": int(u.get("prompt_tokens") or 0), "output_tokens": int(u.get("completion_tokens") or 0)}
        self._count(usage)
        self.messages.append({"role": "assistant", "content": text})
        act, err = parse_action(text, self.tool_names)
        cid = "a%d" % len(self.messages)
        if act is None:
            return Turn(texts=[], calls=[ToolCall(cid, "", {}, error="출력 형식 오류: %s — 출력: %s" % (err, text[:200]))],
                        stop=STOP_TOOL_USE, usage=usage)
        return Turn(texts=[act["thought"]] if act["thought"] else [], calls=[ToolCall(cid, act["tool"], act["input"])],
                    stop=STOP_TOOL_USE, usage=usage)

    def add_tool_results(self, results: List[ToolResult]) -> None:
        # 단계 형식에서는 다음 단계를 start() 로 새로 연다. 대화형으로 쓸 때만 이어 붙인다.
        for r in results:
            self.messages.append({"role": "user", "content": ("ERROR: " if r.is_error else "RESULT: ") + r.content})


__all__ = ["JsonActionBackend", "STOP_END"]
