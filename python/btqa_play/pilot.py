"""LlmPilot — 비전투를 LLM 이 플레이하는 수동 tool use 루프 (PLAY_PROTOCOL 5장). LLM 은 사용자가 고른다.

- LLM 은 `btqa_play.llm.LlmBackend` (Claude API / OpenAI 호환 로컬·클라우드). 안내 docs/PLAY_SDK.md
- 시스템 프롬프트 = SDK 공통 규칙(도구 사용 방식만) + 게임 안내(게임 선언 → 외부 스펙) + 클라이언트 추가 규칙 (prompt.py)
- 도구 = 게임이 지원하는 행동, 설명·스키마는 게임 선언 → 외부 스펙 → 일반형 (tools.build_tools, spec.py). 프롬프트·도구는 세션 내내 고정 (Claude 는 프롬프트 캐시)
- 목표 = PilotConfig.goal → pilot_guide.default_goal (게임 → 스펙) → SDK 기본 목표
- 진척 = progress_keys (게임 → 스펙) 값의 변화, 없으면 관찰 전체(VOLATILE_KEYS 제외)
- 전투는 도구 실행(PlayClient.act) 안에서 학습된 BT·응답기가 끝내고, 요약만 도구 결과에 실린다
- LLM 이 말한 텍스트는 pilot_note(thought) 로 게임 화면에 표시 (play v0.3, 아니면 기록만)
- 상한: 행동 수, API 호출 수, 진척 없음(N 행동 동안 진척 서명 변화 없음)
- 결과: PilotRun (끝난 이유, finish 요약, 전투, 토큰 사용량, 대화 기록) — 재현은 게임 get_log 의 행동열로

anthropic SDK 는 AnthropicBackend 안에서만 import → openai 백엔드·나머지는 Python 3.8 표준 라이브러리로 돈다.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence, Union

from btqa_play.client import PlayClient
from btqa_play.llm.base import STOP_MAX_TOKENS, STOP_PAUSE, STOP_REFUSAL, LlmBackend, ToolResult
from btqa_play.prompt import build_system_prompt, default_goal
from btqa_play.spec import PlaySpec, as_spec, effective_progress_keys
from btqa_play.tools import GameTools, build_tools, compact_observation, to_text

DEFAULT_MODEL = "claude-opus-5-5"


@dataclass
class PilotConfig:
    model: str = DEFAULT_MODEL        # backend 를 안 줄 때(AnthropicBackend) 만 쓴다
    effort: str = "low"
    max_tokens: int = 8000
    max_actions: int = 150            # 게임 행동 수 상한
    max_calls: int = 250              # LLM 호출 상한
    no_progress_actions: int = 25     # 이만큼 행동하는 동안 진척 없으면 중단
    goal: Optional[str] = None        # None = pilot_guide.default_goal (게임 → 스펙) → SDK 기본
    extra_rules: Sequence[str] = ()   # 시스템 프롬프트에 덧붙일 규칙 (게임 안내 뒤)
    inspect_depth: int = 4            # inspect_ui 계층 깊이

@dataclass
class PilotRun:
    end_reason: str = ""
    llm: str = ""
    finish: Optional[Dict[str, Any]] = None
    actions: int = 0
    api_calls: int = 0
    combats: List[Dict[str, Any]] = field(default_factory=list)
    usage: Dict[str, int] = field(default_factory=dict)
    seconds: float = 0.0
    last_observation: Dict[str, Any] = field(default_factory=dict)
    transcript: List[Dict[str, Any]] = field(default_factory=list)   # (assistant 텍스트·도구 호출·결과 요약)

    def to_dict(self) -> Dict[str, Any]:
        return {k: getattr(self, k) for k in ("end_reason", "llm", "finish", "actions", "api_calls", "combats", "usage",
                                              "seconds", "last_observation", "transcript")}


#: progress_keys 가 없을 때 진척 비교에서 빼는 관찰 필드 — 행동할 때마다 바뀌어 진척이 아닌 것 (PLAY_PROTOCOL 4.5)
VOLATILE_KEYS = ("type", "seq", "game_time", "events", "entities", "hidden_included", "player.pos")


def _get(obs: Mapping[str, Any], path: str) -> Any:
    cur: Any = obs
    for part in path.split("."):
        if not isinstance(cur, Mapping):
            return None
        cur = cur.get(part)
    return cur


def _drop(obs: Mapping[str, Any], paths: Sequence[str]) -> Dict[str, Any]:
    out = json.loads(json.dumps(obs, default=str))
    for path in paths:
        parts = path.split(".")
        cur = out
        for part in parts[:-1]:
            cur = cur.get(part) if isinstance(cur, dict) else None
        if isinstance(cur, dict):
            cur.pop(parts[-1], None)
    return out


def progress_signature(obs: Mapping[str, Any], keys: Optional[Sequence[str]] = None) -> str:
    """진척 판단용 서명. keys(관찰 경로 목록, 스펙·게임 progress_keys) 가 있으면 그 값만,
    없으면 관찰 전체에서 VOLATILE_KEYS 를 뺀 것. 서명이 바뀌면 진척으로 본다."""
    if keys:
        return json.dumps([_get(obs, k) for k in keys], ensure_ascii=False, sort_keys=True, default=str)
    return json.dumps(_drop(obs, VOLATILE_KEYS), ensure_ascii=False, sort_keys=True, default=str)


class LlmPilot:
    def __init__(self, client: PlayClient, responder: Any = None, config: Optional[PilotConfig] = None,
                 api_client: Any = None, backend: Optional[LlmBackend] = None,
                 spec: Union[None, PlaySpec, Mapping[str, Any], str] = None) -> None:
        """client 는 hello·describe 를 마친 상태. responder = 전투 trigger 응답기 (None = 바꾸지 않음).
        spec = 외부 플레이 스펙 (PlaySpec·dict·파일 경로) — 게임이 선언하지 않은 설명·스키마·안내를 채운다."""
        self.client = client
        self.cfg = config or PilotConfig()
        self.spec = as_spec(spec)
        self.tools = GameTools(client, responder, inspect_depth=self.cfg.inspect_depth)
        if backend is None:   # 하위 호환: 백엔드를 안 주면 Claude API (api_client = 가짜 주입 가능)
            from btqa_play.llm.anthropic_backend import AnthropicBackend
            backend = AnthropicBackend(model=self.cfg.model, effort=self.cfg.effort, max_tokens=self.cfg.max_tokens,
                                       api_client=api_client)
        self.backend = backend

    def run(self, first_observation: Mapping[str, Any]) -> PilotRun:
        cfg, llm = self.cfg, self.backend
        desc = self.client.description
        intro = {
            "goal": cfg.goal or default_goal(desc, self.spec),
            "game": "%s %s" % (desc.get("game"), desc.get("game_version")),
            "quests_in_game": desc.get("quests"),
            "observation": compact_observation(first_observation),
        }
        llm.start(build_system_prompt(desc, self.spec, cfg.extra_rules), build_tools(desc, self.spec), to_text(intro))
        keys = effective_progress_keys(desc, self.spec)
        run = PilotRun(llm=llm.label)
        t0 = time.time()
        last_sig, stale = progress_signature(first_observation, keys), 0
        run.last_observation = dict(first_observation)

        while True:
            if run.api_calls >= cfg.max_calls:
                run.end_reason = "max_calls"
                break
            turn = llm.complete()
            run.api_calls += 1
            if turn.texts:
                text = "\n".join(turn.texts)
                run.transcript.append({"assistant": text})
                self.client.note(text)   # v0.3 게임이면 화면에 표시, 아니면 클라이언트 기록만

            if turn.stop == STOP_PAUSE:
                continue
            if turn.stop in (STOP_REFUSAL, STOP_MAX_TOKENS) and not turn.calls:
                run.end_reason = "stop_" + turn.stop
                break
            if not turn.calls:
                # 도구 호출은 보장되지 않는다 → 한 번 상기시키고 계속
                llm.add_user("도구로 다음 행동을 하거나 finish 를 불러라.")
                run.transcript.append({"nudge": True})
                continue

            results: List[ToolResult] = []
            for call in turn.calls:
                if call.error:   # 인자 파싱 실패 → 실행하지 않고 알려 준다
                    out, is_err = {"error": call.error}, True
                else:
                    try:
                        out = self.tools.execute(call.name, call.input)
                        is_err = "error" in out
                    except Exception as e:   # PlayError(전투 중 error·연결 끊김) → 세션 종료
                        run.end_reason = "game_error: %s" % e
                        out, is_err = {"error": str(e)}, True
                run.transcript.append({"tool": call.name, "input": call.input,
                                       "status": out.get("status"), "reason": out.get("reason"),
                                       "error": out.get("error"), "combats": out.get("combats")})
                results.append(ToolResult(call.id, to_text(out), is_err))
                if run.end_reason:
                    break
            llm.add_tool_results(results)

            run.actions = self.tools.actions
            if self.tools.last_observation:
                run.last_observation = self.tools.last_observation
            if run.end_reason:
                break
            if self.tools.finished is not None:
                run.finish = self.tools.finished
                run.end_reason = "finish"
                break
            if run.actions >= cfg.max_actions:
                run.end_reason = "max_actions"
                break
            sig = progress_signature(run.last_observation, keys)
            if sig != last_sig:
                last_sig, stale = sig, 0
            else:
                stale += sum(1 for c in turn.calls if c.name not in ("finish", "inspect_ui"))
                if stale >= cfg.no_progress_actions:
                    run.end_reason = "no_progress"
                    break

        run.usage = dict(llm.usage)
        run.combats = [c.summary() for c in self.tools.combats]
        run.seconds = round(time.time() - t0, 1)
        return run
