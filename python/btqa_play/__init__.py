"""BT-QA Play SDK — LLM 이 게임의 비전투를 플레이하고 전투는 학습된 BT 가 맡는 플레이 모드의 **클라이언트 쪽** 키트.

계약 docs/PLAY_PROTOCOL.md (play v0.3). 게임 쪽(서버)은 게임이 직접 구현한다.
표준 라이브러리만 쓴다 (Python ≥ 3.8). anthropic 백엔드만 anthropic SDK(Python ≥ 3.10)를 그 안에서 import.

    PlayClient      프로토콜 클라이언트 (act 중 전투 메시지 수집·trigger 즉답)
    PlaySpec        외부 플레이 스펙 (btqa-play-spec/1) — 게임이 아직 선언하지 않은 설명·스키마·안내·진척 기준
    build_tools     play_description.actions → LLM 도구 (게임 선언 → 외부 스펙 → 일반형). SDK 에 게임 내용 없음
    GameTools       도구 호출 실행기
    LlmPilot        LLM tool use 루프 (LlmBackend: anthropic / openai 호환)
    PilotRelay      대화형 에이전트(Claude Code 등)가 직접 플레이어 — API 없음
    build_system_prompt  SDK 공통 규칙(도구 사용 방식만) + pilot_guide (게임 → 스펙)
    actions / StepRecorder  단계 형식 (지금 관찰 + 직전 결과 → JSON 행동 1개) · 단계 기록 (작은 로컬 모델·증류)
    evaluate.run_batch      여러 seed 평가 → summary (기준선·증류 비교)
    dataset.build_dataset   단계 기록 → 학습용 train/val (학습 자체는 선택 패키지 btqa_distill)
    review                  교사 라벨 검토 (실패한 정답·진척 없는 반복·불확실 → 해소)
    suite.run_suite/compare 평가 스위트 (본 조건·강건성·안 본 조건) + 채택 판정 — SFT 이후(DPO·RFT) 판정
    branch.Brancher         같은 상태에서 후보 행동 시험 (snapshot 또는 seed+prefix 재생)

전투 응답기(trigger → params)는 SDK 밖이다: 학습 결과 형식에 묶이므로 쓰는 쪽이 callable 로 넘긴다.
"""
from btqa_play.actions import action_schema, action_system_prompt, action_text, parse_action, step_input, with_goal
from btqa_play.client import (BOSS_MISMATCH_CODES, NOTE_KINDS, PROTOCOL, PROTOCOL_VERSION, ActResult, Combat, PlayClient,
                              PlayError, TriggerResponder, parse_endpoint)
from btqa_play.pilot import LlmPilot, PilotConfig, PilotRun, progress_signature
from btqa_play.prompt import CORE_PROMPT, DEFAULT_GOAL, build_system_prompt, default_goal
from btqa_play.dataset import DATASET_FORMAT, build_dataset
from btqa_play.evaluate import parse_seeds, run_batch
from btqa_play.record import STEP_FORMAT, StepRecorder, grounding_issues, read_steps
from btqa_play.relay import DEFAULT_RELAY_PORT, PilotRelay, relay_step
from btqa_play.branch import Brancher, Episode
from btqa_play.review import label_flags, resolve, review_list
from btqa_play.suite import SUITE_FORMAT, compare, load_suite, run_suite, validate_suite
from btqa_play.spec import SPEC_FORMAT, PlaySpec, SpecError, as_spec, validate_spec
from btqa_play.tools import (META_TOOLS, GameTools, build_tools, compact_observation, to_text, tool_sources,
                             validate_description)

__version__ = "0.5.0"   # SDK 레포 버전 (btqa_game 과 같이 올린다)

__all__ = [
    "PROTOCOL", "PROTOCOL_VERSION", "NOTE_KINDS", "BOSS_MISMATCH_CODES",
    "PlayClient", "PlayError", "ActResult", "Combat", "TriggerResponder", "parse_endpoint",
    "SPEC_FORMAT", "PlaySpec", "SpecError", "as_spec", "validate_spec",
    "META_TOOLS", "build_tools", "tool_sources", "validate_description", "GameTools", "compact_observation", "to_text",
    "CORE_PROMPT", "DEFAULT_GOAL", "build_system_prompt", "default_goal",
    "LlmPilot", "PilotConfig", "PilotRun", "progress_signature",
    "PilotRelay", "relay_step", "DEFAULT_RELAY_PORT",
    "action_schema", "action_system_prompt", "action_text", "parse_action", "step_input", "with_goal",
    "STEP_FORMAT", "StepRecorder", "grounding_issues", "read_steps",
    "DATASET_FORMAT", "build_dataset", "run_batch", "parse_seeds",
    "Brancher", "Episode", "label_flags", "resolve", "review_list",
    "SUITE_FORMAT", "compare", "load_suite", "run_suite", "validate_suite",
    "__version__",
]
