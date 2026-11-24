"""LLM 백엔드 — 사용자가 자기 LLM 을 고른다 (SDK 가이드 docs/PLAY_SDK.md 4장).

  anthropic : Claude API (anthropic SDK, Python ≥ 3.10 venv, 종량제)
  openai    : OpenAI 호환 /v1/chat/completions — 로컬 LLM(Ollama·LM Studio·llama.cpp·vLLM) 또는 호환 클라우드
  (API 를 아예 안 쓰려면 LLM 백엔드 대신 `python -m btqa_play relay` — Claude Code 같은 대화형 에이전트가 직접 플레이)

설정 = `llm.json`(저장소에 올리지 않음, 예시 `llm.example.json`) + CLI 인자(우선).
키는 파일에 쓰지 않는다 — 파일에는 키가 든 **환경변수 이름**(`api_key_env`)만.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

from btqa_play.llm.base import LlmBackend, ToolCall, ToolResult, Turn

BACKENDS = ("anthropic", "openai")
CONFIG_KEYS = ("llm", "model", "base_url", "api_key_env", "effort", "max_tokens", "temperature", "timeout", "extra_body")
DEFAULT_CONFIG_FILE = "llm.json"


def load_config(path: Optional[str] = None, overrides: Optional[Mapping[str, Any]] = None) -> Dict[str, Any]:
    """llm.json(있으면) 위에 CLI 값(None 이 아닌 것)을 덮는다. 키 값이 들어 있으면 거부."""
    cfg: Dict[str, Any] = {}
    p = Path(path) if path else Path(DEFAULT_CONFIG_FILE)
    if p.exists():
        cfg = json.loads(p.read_text(encoding="utf-8"))
        bad = [k for k in cfg if k not in CONFIG_KEYS and not k.startswith("_")]
        if bad:
            raise ValueError("%s: 모르는 키 %s (허용 %s)" % (p, bad, CONFIG_KEYS))
        if any(k in cfg for k in ("api_key", "key", "token")) or str(cfg.get("api_key_env", "")).startswith("sk-"):
            raise ValueError("%s: 키 값을 파일에 쓰지 않는다 — api_key_env 에 환경변수 **이름**만" % p)
    elif path:
        raise FileNotFoundError(path)
    for k, v in (overrides or {}).items():
        if v is not None:
            cfg[k] = v
    return cfg


def make_backend(cfg: Mapping[str, Any]) -> LlmBackend:
    kind = cfg.get("llm")
    if kind not in BACKENDS:
        raise ValueError("LLM 을 골라야 한다: --llm %s 또는 llm.json 의 \"llm\" (안내 docs/PLAY_SDK.md 4장)" % "|".join(BACKENDS))
    if kind == "anthropic":
        from btqa_play.llm.anthropic_backend import DEFAULT_MODEL, AnthropicBackend
        return AnthropicBackend(model=cfg.get("model") or DEFAULT_MODEL, effort=cfg.get("effort") or "low",
                                max_tokens=int(cfg.get("max_tokens") or 8000), api_key_env=cfg.get("api_key_env"))
    from btqa_play.llm.openai_compat import DEFAULT_BASE_URL, OpenAICompatBackend
    return OpenAICompatBackend(model=cfg.get("model") or "", base_url=cfg.get("base_url") or DEFAULT_BASE_URL,
                               api_key_env=cfg.get("api_key_env"), max_tokens=int(cfg.get("max_tokens") or 4000),
                               temperature=cfg.get("temperature"), timeout=float(cfg.get("timeout") or 600.0),
                               extra_body=cfg.get("extra_body"))


__all__ = ["LlmBackend", "ToolCall", "ToolResult", "Turn", "BACKENDS", "load_config", "make_backend"]
