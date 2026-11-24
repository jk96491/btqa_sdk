"""LLM 플레이어 시스템 프롬프트 = SDK 공통 규칙 + 게임 안내.

SDK 공통 규칙(CORE_PROMPT)은 **도구 사용 방식만** 말한다 — 장르·화면·아이템 같은 게임 개념은 없다.
게임 안내 = pilot_guide (게임 선언 → 외부 스펙 순, 키마다, spec.effective_guide):
    system_prompt  관찰을 읽는 법·이 게임에서 지켜야 할 큰 규칙 (여러 줄 문단)
    about          게임 한 줄 소개
    rules          플레이 요령 목록
    default_goal   목표를 따로 주지 않았을 때 LLM 목표
+ 쓰는 쪽 extra_rules.

세션 내내 같은 문자열이어야 한다 (프롬프트 캐시).
"""
from __future__ import annotations

from typing import Any, Mapping, Optional, Sequence

from btqa_play.spec import PlaySpec, effective_guide

CORE_PROMPT = """너는 게임 QA 플레이어다. 게임을 도구로 직접 플레이한다. 화면 대신 구조화된 관찰(JSON)을 받는다.

규칙:
- 세계는 네가 행동할 때만 흐른다. 서두를 필요 없다.
- 행동은 반드시 도구로 한다. 한 번에 도구 하나씩 부르고 결과(status·reason·observation)를 보고 다음을 정한다.
- 도구를 부르기 전에 왜 그 행동을 하는지 한 문장으로 말한다 (게임 화면에 플레이어 생각으로 표시될 수 있다).
- 실패하면 reason 을 읽고 방법을 바꾼다. 같은 실패를 되풀이하지 않는다.
- 전투는 네가 하지 않는다. 전투가 일어나면 학습된 전투 AI 가 싸우고 결과가 combats 로 온다.
- 목표를 이뤘거나, 같은 시도를 반복해도 진척이 없으면 finish 를 부른다. 게임 버그로 보이는 것은 blocked 에 구체적으로 적는다.
- 말은 짧게."""

#: 목표가 어디에도 없을 때 (게임 개념 없이)
DEFAULT_GOAL = "게임이 제시하는 목표를 진행할 수 있는 데까지 진행하라."


def build_system_prompt(description: Mapping[str, Any], spec: Optional[PlaySpec] = None,
                        extra_rules: Optional[Sequence[str]] = None) -> str:
    g = effective_guide(description, spec)
    parts = [CORE_PROMPT]
    sp = g.get("system_prompt")
    if isinstance(sp, str) and sp.strip():
        parts.append(sp.strip())
    about = g.get("about")
    rules = [r for r in (g.get("rules") or []) if isinstance(r, str) and r.strip()]
    rules += [r for r in (extra_rules or []) if r and r.strip() and r not in rules]
    if (isinstance(about, str) and about.strip()) or rules:
        lines = ["%s 안내:" % (description.get("game") or "이 게임")]
        if isinstance(about, str) and about.strip():
            lines.append(about.strip())
        lines += ["- " + r.strip() for r in rules]
        parts.append("\n".join(lines))
    return "\n\n".join(parts)


def default_goal(description: Mapping[str, Any], spec: Optional[PlaySpec] = None, fallback: str = DEFAULT_GOAL) -> str:
    goal = effective_guide(description, spec).get("default_goal")
    return goal.strip() if isinstance(goal, str) and goal.strip() else fallback


__all__ = ["CORE_PROMPT", "DEFAULT_GOAL", "build_system_prompt", "default_goal"]
