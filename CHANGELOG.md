# 변경 이력

## 0.4.0 — 2026-11-24

**작은 로컬 LLM(CPU 추론)·증류용 단계 형식** (`btqa_play`). 밸런스 프로토콜 1.4·플레이 0.3 그대로, `btqa_game`·Unity 패키지 코드 변화 없음 (버전만).

- `actions.py` 단계 형식: 한 단계 = [고정 system(규칙+목표+도구+출력 형식), 지금 관찰 + 직전 행동 결과] → JSON 행동 1개 `{"thought","tool","input"}`. 대화 기록 없음. `action_schema` = 출력 JSON Schema (도구별 anyOf)
- `LlmPilot(config=PilotConfig(stateless=True))`: 단계마다 새 대화, `PilotRun.invalid_outputs`·`latency_s`
- `llm/json_action.py: JsonActionBackend` (`tool_mode: "json"`): 서버 function calling 대신 `response_format: json_schema` 로 출력 강제 (Ollama·llama.cpp), OpenAI 호환·표준 라이브러리
- `record.py: StepRecorder` 단계 기록 JSONL (`btqa-play-step/1`: 학생 입력 그대로, 목표, 결과, 토큰, 지연) + `grounding_issues` (입력에 없는 id·근거 없는 좌표 표시 — 교사가 학생이 못 보는 정보로 판단한 샘플 거르기)
- `PilotRelay(record=...)`: 교사 기록 모드 — 응답 = 학생과 같은 다음 단계 입력만, `step view`
- `observation_drop` (스펙·게임 선언 4.2.1): LLM 에게 보이기 전에 뺄 관찰 경로 (`a.b`, 목록 원소마다, `name[k=v]` 필터) — 입력 길이
- CLI: `play --tool-mode json --stateless --record F`, `relay --record F`
- `evaluate.py: run_batch` 여러 seed 평가 → summary.json (완료율·형식 오류율·호출당 시간·판별 토큰), `python -m btqa_play batch`
- `dataset.py: build_dataset` 단계 기록 → 학습 train/val (`btqa-play-sft/1`): source·형식 오류·grounding·중복 거르기, 판 단위 분할, `python -m btqa_play dataset`
- **선택 패키지 `btqa_distill`** (SDK 에서 유일하게 외부 의존성: torch·transformers·peft, extra `distill`): `TrainConfig`, `train_lora`(assistant 에만 loss, bf16 + LoRA fp32), `export`(merge → llama.cpp GGUF → 양자화 → Ollama, 기본 CPU 전용 `num_gpu 0`), `python -m btqa_distill check|train|merge|gguf|ollama|cpu-variant`. 가이드 `docs/DISTILL_SDK.md`
- 모델 파일 커밋 금지: `.gitignore` 모델 규칙 + `tests/test_no_model_files.py`
- `LlmPilot(on_step=...)`: 단계마다 콜백 (화면·콘솔 표시용)
- 수정: 판마다 토큰 사용량을 따로 (백엔드를 여러 판에 쓰면 누적되던 문제)

## 0.3.0 — 2026-11-24

**플레이 모드 클라이언트 `btqa_play` 추가** (밸런스 프로토콜 1.4 그대로, `btqa_game`·Unity 패키지 코드 변화 없음 — 버전만).

- 계약 `docs/PLAY_PROTOCOL.md` 를 SDK 로 옮김 (원본). play v0.3 + **선택 필드** 4.2.1: `actions[].description`·`actions[].schema`(LLM 도구 설명·인자 JSON Schema), `pilot_guide{system_prompt, about, rules, default_goal}`, `progress_keys`. hello 버전 그대로 (보내지 않는 게임·읽지 않는 클라이언트 모두 그대로 동작)
- `btqa_play`: `PlayClient`(전투 메시지 수집·trigger 즉답·pilot_note), **SDK 에 게임 내용 없음**: `PlaySpec`(외부 플레이 스펙 `btqa-play-spec/1`), `build_tools`(게임 선언 → 외부 스펙 → 일반형)·`GameTools`·`validate_description`·`tool_sources`, `build_system_prompt`(SDK 공통 규칙 = 도구 사용 방식만 + `pilot_guide`), 진척 = `progress_keys` (없으면 관찰 전체 − 매번 바뀌는 필드), `LlmBackend`(`AnthropicBackend`·`OpenAICompatBackend`, `llm.json`), `LlmPilot`, `PilotRelay`(확장 지점 `prepare`·`run_meta`), `python -m btqa_play tools|spec|play|relay|step` (`--spec`)
- 출처: AI_QA `play_agent` (2026-11-23) 의 클라이언트·도구·루프·백엔드를 게임 독립으로 옮김. 전투 로드아웃·응답기(학습 결과 형식)는 SDK 밖
- 가이드 `docs/PLAY_SDK.md`, 테스트 `tests/test_play.py` (가짜 플레이 게임·가짜 LLM)

## 0.2.0 — 2026-11-23

프로토콜 1.4 (부 버전, 1.3 과 호환).

- `run_episode.initial_state`: 판 시작 상태(체력·자원·물약 비율 등)를 요청으로 정한다. 게임은 `describe.initial_state` 로 받을 키·범위·default 를 선언 (이름 = `triggers.state_features` 중 하나). 생략·빈 객체 = 1.3 과 같은 판. 모르는 키 `unknown_param`, 범위 밖 `param_out_of_range`, 숫자 아님 `bad_request`. 선언하지 않은 게임은 요청을 `unknown_param` 으로 거부
- 용도: 학습 쪽 컨텍스트 밴딧(판 시작 상황마다 다른 블록 가중치) — 플레이 모드 필드 전투가 가득 차지 않은 상태로 시작하는 것을 학습 판에서 재현
- Python `InitialStateDecl`, `GameDecl.initial_state`·`resolve_initial_state`, `EpisodeRequest.initial_state` / Unity `InitialStateDecl`, `GameDecl.InitialState`·`ResolveInitialState`, `EpisodeRequest.InitialState`
- 적합성: 선언했으면 default=생략 동일·결정성·트리거 0 반영(권장)·오류 코드, 선언 안 했으면 요청 거부 (1.4 이상 필수)
- 예제 `minimal_arena`: `hp_ratio`·`potions_ratio` 시작 상태

## 0.1.0 — 2026-11-23

첫 판. 프로토콜 1.3.

- 계약 `docs/PROTOCOL.md` v1.3: `classes[].blocks` 추가, **학습 대상 = 블록 가중치뿐**, `run_episode.triggers` 키 = `describe.triggers.type`
- Python `btqa_game`: `GameDecl`·`ClassDecl`·`ParamDecl`·`BlockDecl`·`TriggerDecl`, `EpisodeHost`, `GameSession`, `QaGameServer`
- Python `btqa_game.conformance`: 적합성 검사 (`python -m btqa_game check --game HOST:PORT`, 표준 라이브러리만)
- Python `btqa_game.examples.minimal_arena`: 블록 이름이 다른 최소 예제 게임
- Unity `com.btqa.game` (`Btqa.Game`): `Wire`·`GameDecl`·`IEpisodeHost`·`TriggerChannel`·`QaSession`·`QaServer`
- 처음 적용한 게임: game_sim 0.5.0 (Python), QaArena 0.7.0 (Unity) — 판 결과는 SDK 도입 전과 바이트 단위 동일
