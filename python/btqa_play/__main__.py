"""python -m btqa_play <명령> — 플레이 모드 클라이언트 SDK 명령줄.

    tools  --game HOST:PORT [--spec F]      게임 선언(+외부 스펙)으로 만든 LLM 도구·출처·시스템 프롬프트·선언 검사
    spec   F                                외부 플레이 스펙 파일 형식 검사 (게임 없이)
    play   --game HOST:PORT --llm openai --model qwen2.5:14b   각자 고른 LLM 이 플레이 (로드아웃 없음 = 게임 default BT)
    relay  --game HOST:PORT [--port 7899]   대화형 에이전트가 플레이어 (API 없음). 이후 step 으로 한 행동씩
    step   <tool> ['<json>'] [--think "..."] [--kind plan]   relay 에 도구 호출 1개

tools·play·relay 의 --spec F: 게임이 아직 선언하지 않은 행동 설명·스키마·안내를 외부 스펙 파일(btqa-play-spec/1)로 채운다.
전투 로드아웃·학습된 응답기가 필요하면 이 SDK 위에 쓰는 쪽 클라이언트를 만든다 (예: AI_QA 의 play_agent).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import List, Optional

from btqa_play.client import PlayClient
from btqa_play.prompt import build_system_prompt, default_goal
from btqa_play.spec import PlaySpec, SpecError, as_spec, validate_spec
from btqa_play.relay import DEFAULT_RELAY_PORT, PilotRelay, relay_step
from btqa_play.tools import build_tools, tool_sources, validate_description


def _utf8() -> None:
    for s in (sys.stdout, sys.stderr):
        try:
            s.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
        except Exception:
            pass


def cmd_spec(a: argparse.Namespace) -> int:
    problems = validate_spec(json.loads(Path(a.file).read_text(encoding="utf-8")))
    print("%s: %s" % (a.file, "통과" if not problems else "문제 %d개" % len(problems)))
    for p in problems:
        print("  문제:", p)
    return 1 if problems else 0


def _spec(a: argparse.Namespace) -> PlaySpec:
    try:
        return as_spec(getattr(a, "spec", None))
    except (SpecError, OSError, ValueError) as e:
        raise SystemExit("스펙: %s" % e)


def cmd_tools(a: argparse.Namespace) -> int:
    spec = _spec(a)
    with PlayClient.connect(a.game) as c:
        c.hello()
        desc = c.describe()
    print("게임: %s %s (play %s)" % (desc.get("game"), desc.get("game_version"), c.protocol_version))
    print("외부 스펙:", spec.source or "없음")
    for name, src in tool_sources(desc, spec).items():
        print("  %-10s 설명/스키마 출처 = %s" % (name, src))
    problems = validate_description(desc, spec)
    print("선언 검사:", "통과" if not problems else "")
    for p in problems:
        print("  문제:", p)
    if a.json:
        print(json.dumps(build_tools(desc, spec), ensure_ascii=False, indent=2))
    print("\n목표:", default_goal(desc, spec))
    print("\n--- 시스템 프롬프트 ---\n" + build_system_prompt(desc, spec))
    return 1 if problems else 0


def cmd_play(a: argparse.Namespace) -> int:
    from btqa_play.llm import load_config, make_backend
    from btqa_play.pilot import LlmPilot, PilotConfig

    try:
        backend = make_backend(load_config(a.llm_config, {"llm": a.llm, "model": a.model, "base_url": a.base_url,
                                                          "api_key_env": a.api_key_env, "effort": a.effort}))
    except (ValueError, FileNotFoundError) as e:
        print("LLM 설정:", e, file=sys.stderr)
        return 2
    with PlayClient.connect(a.game) as c:
        c.hello()
        c.describe()
        obs = c.reset(seed=a.seed, cls=a.cls)
        run = LlmPilot(c, config=PilotConfig(max_actions=a.max_actions, max_calls=a.max_calls, goal=a.goal),
                       backend=backend, spec=_spec(a)).run(obs)
        log = c.get_log()
        notes, pilot_notes = list(c.notes), list(c.pilot_notes)
    out = Path(a.out) / ("play-" + time.strftime("%Y%m%d-%H%M%S"))
    out.mkdir(parents=True, exist_ok=True)
    meta = {"seed": a.seed, "class": a.cls, "notes": notes, "pilot_notes": pilot_notes}
    (out / "run.json").write_text(json.dumps(dict(meta, **run.to_dict()), ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    (out / "play_log.json").write_text(json.dumps(log, ensure_ascii=False), encoding="utf-8")
    print("끝: %s | 행동 %d, LLM %d회 | 기록 %s" % (run.end_reason, run.actions, run.api_calls, out))
    return 0


def cmd_relay(a: argparse.Namespace) -> int:
    PilotRelay(a.game, seed=a.seed, cls=a.cls, port=a.port, out=a.out, spec=_spec(a)).serve()
    return 0


def cmd_step(a: argparse.Namespace) -> int:
    print(relay_step(a.tool, json.loads(a.input or "{}"), port=a.port, thought=a.think, kind=a.kind))
    return 0


def main(argv: Optional[List[str]] = None) -> int:
    _utf8()
    ap = argparse.ArgumentParser(prog="python -m btqa_play", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("tools", help="게임 선언 → LLM 도구·시스템 프롬프트 미리 보기 + 선언 검사")
    p.add_argument("--game", required=True)
    p.add_argument("--json", action="store_true", help="도구 정의 전체 출력")
    p.add_argument("--spec", help="외부 플레이 스펙 파일 (btqa-play-spec/1)")
    p.set_defaults(fn=cmd_tools)

    p = sub.add_parser("spec", help="외부 플레이 스펙 파일 형식 검사")
    p.add_argument("file")
    p.set_defaults(fn=cmd_spec)

    p = sub.add_parser("play", help="각자 고른 LLM 이 플레이 (로드아웃 없이)")
    p.add_argument("--game", required=True)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--cls")
    p.add_argument("--goal")
    p.add_argument("--llm", choices=["anthropic", "openai"])
    p.add_argument("--llm-config")
    p.add_argument("--model")
    p.add_argument("--base-url")
    p.add_argument("--api-key-env")
    p.add_argument("--effort", choices=["low", "medium", "high", "xhigh", "max"])
    p.add_argument("--max-actions", type=int, default=150)
    p.add_argument("--max-calls", type=int, default=250)
    p.add_argument("--out", default="reports")
    p.add_argument("--spec", help="외부 플레이 스펙 파일")
    p.set_defaults(fn=cmd_play)

    p = sub.add_parser("relay", help="대화형 에이전트가 플레이어가 되는 중계기 (API 호출 없음)")
    p.add_argument("--game", required=True)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--cls")
    p.add_argument("--port", type=int, default=DEFAULT_RELAY_PORT)
    p.add_argument("--out", default="reports")
    p.add_argument("--spec", help="외부 플레이 스펙 파일")
    p.set_defaults(fn=cmd_relay)

    p = sub.add_parser("step", help="relay 에 도구 호출 1개 (tool=intro 처음 관찰·도구·규칙, note 메모만)")
    p.add_argument("tool")
    p.add_argument("input", nargs="?")
    p.add_argument("--think")
    p.add_argument("--kind", default="thought", choices=["thought", "plan", "finding"])
    p.add_argument("--port", type=int, default=DEFAULT_RELAY_PORT)
    p.set_defaults(fn=cmd_step)

    a = ap.parse_args(argv)
    return a.fn(a)


if __name__ == "__main__":
    sys.exit(main())
