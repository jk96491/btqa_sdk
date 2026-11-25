"""python -m btqa_play <명령> — 플레이 모드 클라이언트 SDK 명령줄.

    tools  --game HOST:PORT [--spec F]      게임 선언(+외부 스펙)으로 만든 LLM 도구·출처·시스템 프롬프트·선언 검사
    spec   F                                외부 플레이 스펙 파일 형식 검사 (게임 없이)
    play   --game HOST:PORT --llm openai --model qwen2.5:14b   각자 고른 LLM 이 플레이 (로드아웃 없음 = 게임 default BT)
           작은 로컬 모델 (CPU): --tool-mode json --stateless [--record steps.jsonl]
    relay  --game HOST:PORT [--port 7899]   대화형 에이전트가 플레이어 (API 없음). 이후 step 으로 한 행동씩
           증류 교사 데이터: --record teacher.jsonl (학생과 같은 단계 입력만 보고 판단)
    step   <tool> ['<json>'] [--think "..."] [--kind plan]   relay 에 도구 호출 1개
    batch  --game HOST:PORT --seeds 1-5 --llm openai --model M --tool-mode json --stateless [--record]   여러 seed 평가 → summary.json
    dataset <기록.jsonl ...> --out <폴더>      단계 기록 → 학습용 train/val (증류, 학습은 python -m btqa_distill)

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
from btqa_play.record import StepRecorder
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
                                                          "api_key_env": a.api_key_env, "effort": a.effort,
                                                          "tool_mode": a.tool_mode}))
    except (ValueError, FileNotFoundError) as e:
        print("LLM 설정:", e, file=sys.stderr)
        return 2
    with PlayClient.connect(a.game) as c:
        c.hello()
        c.describe()
        obs = c.reset(seed=a.seed, cls=a.cls)
        rec = StepRecorder(a.record, run="play-%s-s%s" % (time.strftime("%Y%m%d-%H%M%S"), a.seed), source="model",
                           model=backend.label) if a.record else None
        run = LlmPilot(c, config=PilotConfig(max_actions=a.max_actions, max_calls=a.max_calls, goal=a.goal,
                                             stateless=a.stateless or a.record is not None),
                       backend=backend, spec=_spec(a), recorder=rec).run(obs)
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
    PilotRelay(a.game, seed=a.seed, cls=a.cls, port=a.port, out=a.out, spec=_spec(a),
               stateless=a.stateless, record=a.record, student=student_backend(a.student, a.student_url)).serve()
    return 0


def cmd_batch(a: argparse.Namespace) -> int:
    from btqa_play.evaluate import parse_seeds, run_batch
    from btqa_play.llm import load_config, make_backend
    from btqa_play.pilot import PilotConfig
    try:
        cfg = load_config(a.llm_config, {"llm": a.llm, "model": a.model, "base_url": a.base_url, "api_key_env": a.api_key_env,
                                         "effort": a.effort, "tool_mode": a.tool_mode})
        backend = make_backend(cfg)
    except (ValueError, FileNotFoundError) as e:
        print("LLM 설정:", e, file=sys.stderr)
        return 2
    out = Path(a.out) / ("batch-" + time.strftime("%Y%m%d-%H%M%S"))
    with PlayClient.connect(a.game) as c:
        c.hello()
        c.describe()
        s = run_batch(c, backend, parse_seeds(a.seeds), PilotConfig(max_actions=a.max_actions, max_calls=a.max_calls,
                                                                    goal=a.goal, stateless=a.stateless),
                      spec=_spec(a), cls=a.cls, out_dir=out, record=a.record,
                      on_run=lambda r: print("seed %(seed)d: %(end_reason)s | 퀘스트 %(quests_done)d/%(quests_total)d | "
                                             "행동 %(actions)d, 형식 오류 %(invalid_outputs)d | 호출당 %(latency_mean_s).1fs" % r, flush=True))
    print("요약: 전부 완료 %.0f%% | 퀘스트 평균 %.1f/%d | 형식 오류율 %.1f%% | %s" % (
        100 * s["full_clear_rate"], s["quests_done_mean"], s["quests_total"], 100 * s["invalid_rate"], out))
    return 0


def cmd_dataset(a: argparse.Namespace) -> int:
    from btqa_play.dataset import build_dataset
    st = build_dataset(a.inputs, a.out, val_ratio=a.val_ratio, seed=a.seed, sources=tuple(a.sources),
                       keep_ungrounded=a.keep_ungrounded, val_runs=a.val_runs)
    print(json.dumps({k: st[k] for k in ("read", "kept", "dropped", "conflicting_inputs", "thought_variants", "runs", "train", "val", "tools")},
                     ensure_ascii=False))
    print("→", a.out, "(train.jsonl / val.jsonl / stats.json) — git 제외 경로에 둘 것")
    return 0


def student_backend(model: Optional[str], base_url: str):
    """교정 모드 학생 (OpenAI 호환 + JSON 행동 강제). model 없으면 None."""
    if not model:
        return None
    from btqa_play.llm.json_action import JsonActionBackend
    return JsonActionBackend(model, base_url=base_url, max_tokens=300, temperature=0.0)


def cmd_step(a: argparse.Namespace) -> int:
    print(relay_step(a.tool, json.loads(a.input or "{}"), port=a.port, thought=a.think, kind=a.kind, execute=a.execute))
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
    p.add_argument("--tool-mode", choices=["native", "json"], help="openai: json = 출력 JSON 을 스키마로 강제 (작은 로컬 모델)")
    p.add_argument("--stateless", action="store_true", help="단계마다 새 대화 (지금 관찰 + 직전 행동 결과만)")
    p.add_argument("--record", help="단계 기록 JSONL (stateless 를 켠다)")
    p.set_defaults(fn=cmd_play)

    p = sub.add_parser("relay", help="대화형 에이전트가 플레이어가 되는 중계기 (API 호출 없음)")
    p.add_argument("--game", required=True)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--cls")
    p.add_argument("--port", type=int, default=DEFAULT_RELAY_PORT)
    p.add_argument("--out", default="reports")
    p.add_argument("--spec", help="외부 플레이 스펙 파일")
    p.add_argument("--stateless", action="store_true", help="단계 형식: 응답에 학생과 같은 다음 단계 입력만")
    p.add_argument("--record", help="교사 데이터 JSONL (stateless 를 켠다)")
    p.add_argument("--student", help="교정 모드: 학생 모델 이름 (OpenAI 호환, 예 Ollama qa-qwen1.5b-r1) — view 에 학생 제안")
    p.add_argument("--student-url", default="http://localhost:11434/v1")
    p.set_defaults(fn=cmd_relay)

    p = sub.add_parser("batch", help="같은 LLM 으로 여러 seed 평가 → summary.json")
    p.add_argument("--game", required=True)
    p.add_argument("--seeds", default="1-5")
    p.add_argument("--cls")
    p.add_argument("--goal")
    p.add_argument("--llm", choices=["anthropic", "openai"])
    p.add_argument("--llm-config")
    p.add_argument("--model")
    p.add_argument("--base-url")
    p.add_argument("--api-key-env")
    p.add_argument("--effort", choices=["low", "medium", "high", "xhigh", "max"])
    p.add_argument("--tool-mode", choices=["native", "json"])
    p.add_argument("--stateless", action="store_true")
    p.add_argument("--record", action="store_true", help="판별 단계 기록 steps.jsonl (stateless 를 켠다)")
    p.add_argument("--max-actions", type=int, default=150)
    p.add_argument("--max-calls", type=int, default=250)
    p.add_argument("--out", default="reports")
    p.add_argument("--spec")
    p.set_defaults(fn=cmd_batch)

    p = sub.add_parser("dataset", help="단계 기록 JSONL → 학습용 train/val (증류)")
    p.add_argument("inputs", nargs="+")
    p.add_argument("--out", required=True)
    p.add_argument("--val-ratio", type=float, default=0.1)
    p.add_argument("--val-runs", nargs="+", help="검증으로 뺄 판 이름(끝부분 일치, 예 -s5) — 주면 --val-ratio 무작위 대신")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--sources", nargs="+", default=["teacher"], help="쓸 기록 source (기본 teacher)")
    p.add_argument("--keep-ungrounded", action="store_true", help="입력에 근거 없는 샘플도 남긴다 (기본: 뺀다)")
    p.set_defaults(fn=cmd_dataset)

    p = sub.add_parser("step", help="relay 에 도구 호출 1개 (tool=intro 처음 관찰·도구·규칙, note 메모만, view 지금 단계 입력)")
    p.add_argument("tool")
    p.add_argument("input", nargs="?")
    p.add_argument("--think")
    p.add_argument("--kind", default="thought", choices=["thought", "plan", "finding"])
    p.add_argument("--port", type=int, default=DEFAULT_RELAY_PORT)
    p.add_argument("--execute", choices=["student"], help="교정 모드: 정답만 기록하고 게임에는 학생 제안을 실행")
    p.set_defaults(fn=cmd_step)

    a = ap.parse_args(argv)
    return a.fn(a)


if __name__ == "__main__":
    sys.exit(main())
