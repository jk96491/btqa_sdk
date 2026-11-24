"""OpenAI 호환 Chat Completions 백엔드 — 로컬 LLM(Ollama·LM Studio·llama.cpp server·vLLM) 과
OpenAI 호환 클라우드(OpenAI·OpenRouter 등)를 같은 코드로 쓴다.

표준 라이브러리(urllib)만 쓴다 → 프로젝트 venv(Python 3.8)에서 그대로 돈다.
POST {base_url}/chat/completions, 도구 = function calling (tools / tool_calls / role:"tool").
키는 api_key_env 로 지정한 환경변수에서만 읽는다 (로컬 서버는 보통 키가 없다).
"""
from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from typing import Any, Callable, Dict, List, Mapping, Optional

from btqa_play.llm.base import STOP_END, STOP_MAX_TOKENS, STOP_TOOL_USE, LlmBackend, ToolCall, ToolResult, Turn

DEFAULT_BASE_URL = "http://localhost:11434/v1"   # Ollama 기본 (LM Studio = http://localhost:1234/v1)
_STOP = {"tool_calls": STOP_TOOL_USE, "function_call": STOP_TOOL_USE, "stop": STOP_END, "length": STOP_MAX_TOKENS}

Post = Callable[[str, Dict[str, str], Dict[str, Any], float], Dict[str, Any]]


def _http_post(url: str, headers: Dict[str, str], body: Dict[str, Any], timeout: float) -> Dict[str, Any]:
    req = urllib.request.Request(url, data=json.dumps(body, ensure_ascii=False).encode("utf-8"), headers=headers, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        detail = e.read().decode("utf-8", "replace")[:500]
        raise RuntimeError("LLM 서버 HTTP %d: %s" % (e.code, detail)) from None
    except urllib.error.URLError as e:
        raise RuntimeError("LLM 서버에 접속 못 함 (%s): %s — 서버가 떠 있는지, base_url 이 맞는지 확인" % (url, e.reason)) from None


def to_openai_tool(t: Mapping[str, Any]) -> Dict[str, Any]:
    """중립 도구 정의 → OpenAI function 도구. strict 는 서버마다 지원이 달라 넣지 않는다."""
    return {"type": "function", "function": {"name": t["name"], "description": t.get("description", ""),
                                             "parameters": t["input_schema"]}}


class OpenAICompatBackend(LlmBackend):
    def __init__(self, model: str, base_url: str = DEFAULT_BASE_URL, api_key_env: Optional[str] = None,
                 max_tokens: int = 4000, temperature: Optional[float] = None, timeout: float = 600.0,
                 extra_body: Optional[Mapping[str, Any]] = None, post: Optional[Post] = None) -> None:
        super().__init__()
        if not model:
            raise ValueError("openai 백엔드는 model 이 필요하다 (예: qwen2.5:14b, gpt-4.1-mini)")
        self.model, self.base_url = model, base_url.rstrip("/")
        self.label = "openai:%s@%s" % (model, self.base_url)
        self.max_tokens, self.temperature, self.timeout = max_tokens, temperature, timeout
        self.extra_body = dict(extra_body or {})
        self.api_key_env = api_key_env
        self._post = post or _http_post   # 테스트에서 가짜 주입
        self.tools: List[Dict[str, Any]] = []
        self.messages: List[Dict[str, Any]] = []

    def _headers(self) -> Dict[str, str]:
        h = {"Content-Type": "application/json"}
        if self.api_key_env:
            key = os.environ.get(self.api_key_env)
            if not key:
                raise RuntimeError("환경변수 %s 가 비어 있음 (키는 환경변수로만 넣는다, docs/PLAY_SDK.md 4장)" % self.api_key_env)
            h["Authorization"] = "Bearer " + key
        return h

    def start(self, system: str, tools: List[Mapping[str, Any]], first_user: str) -> None:
        self.tools = [to_openai_tool(t) for t in tools]
        self.messages = [{"role": "system", "content": system}, {"role": "user", "content": first_user}]

    def complete(self) -> Turn:
        body: Dict[str, Any] = dict(self.extra_body, model=self.model, messages=self.messages, tools=self.tools,
                                    tool_choice="auto", max_tokens=self.max_tokens)
        if self.temperature is not None:
            body["temperature"] = self.temperature
        resp = self._post(self.base_url + "/chat/completions", self._headers(), body, self.timeout)
        choice = (resp.get("choices") or [{}])[0]
        msg = choice.get("message") or {}
        u = resp.get("usage") or {}
        usage = {"input_tokens": int(u.get("prompt_tokens") or 0), "output_tokens": int(u.get("completion_tokens") or 0)}
        self._count(usage)

        calls: List[ToolCall] = []
        for i, tc in enumerate(msg.get("tool_calls") or []):
            fn = tc.get("function") or {}
            raw = fn.get("arguments")
            cid = tc.get("id") or "call_%d_%d" % (len(self.messages), i)
            tc["id"] = cid   # id 없는 서버도 있다 → 기록에도 같은 id
            try:
                args = raw if isinstance(raw, dict) else json.loads(raw or "{}")
                calls.append(ToolCall(cid, str(fn.get("name")), dict(args)))
            except (ValueError, TypeError) as e:
                calls.append(ToolCall(cid, str(fn.get("name")), {}, error="도구 인자 JSON 오류: %s" % e))
        rec: Dict[str, Any] = {"role": "assistant", "content": msg.get("content") or ""}
        if msg.get("tool_calls"):
            rec["tool_calls"] = msg["tool_calls"]
        self.messages.append(rec)

        text = (msg.get("content") or "").strip()
        stop = STOP_TOOL_USE if calls else _STOP.get(str(choice.get("finish_reason")), STOP_END)
        return Turn(texts=[text] if text else [], calls=calls, stop=stop, usage=usage)

    def add_tool_results(self, results: List[ToolResult]) -> None:
        for r in results:
            self.messages.append({"role": "tool", "tool_call_id": r.call_id,
                                  "content": ("ERROR: " if r.is_error else "") + r.content})

    def add_user(self, text: str) -> None:
        self.messages.append({"role": "user", "content": text})
