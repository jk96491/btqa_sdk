# 변경 이력

## 0.5.0 — 2026-11-26

**SFT 이후(DPO·RFT)를 판정할 수 있는 파이프라인 + 교사 라벨 검토.** 밸런스 프로토콜 1.4 그대로, 플레이 0.3 에 **선택 기능**만 추가 (hello 버전 유지). `btqa_game`·Unity 패키지 코드 변화 없음 (버전만).
AI_QA 첫 시도(2026-11-25)에서 DPO·RFT 가 모두 후퇴한 원인 — 포화된 평가, seed 가 바꾸지 않는 판, 18개 off-policy 쌍, std 로 키운 한 판 신호 — 을 SDK 쪽에서 막는다. 게임 내용은 넣지 않았다 (조건 값·보상·상황 키는 게임 쪽 파일·훅).

- 플레이 프로토콜 0.3 (추가 2, 선택): `play_description.reset_options` + `reset.options` (4.4.1, 안 본 시작 조건), `capabilities.snapshot` + `snapshot`/`restore` (4.11, 같은 상태 분기). `PlayClient.reset(options=)` (선언하지 않은 키는 보내기 전에 `unknown_reset_option`), `reset_option_names`, `supports_snapshot`, `snapshot()`, `restore()`
- `btqa_play.branch`: `Brancher`(snapshot 선언 시 snapshot/restore, 아니면 seed+prefix 재생, 같은 행동 한 번만), `Episode`(seed·class·options·prepare 인자), `prefixes`·`executed_action`(교정 모드의 실제 실행 행동), `call_prepare`(prepare 훅에 조건 인자)
- `btqa_play.review` + relay: 정답으로 기록한 행동이 실패(`label_failed`)하거나 진척 없이 반복(`repeat_no_progress`, progress_keys 서명)하면 응답 `warning` + 기록 `needs_review`. `step --unsure` → 기록 `uncertain`(생각 문장에 글자로 넣지 않음). `approve` 는 경고에서 멈춤. CLI `review`·`resolve`(맞음 `reviewed` / 정답 고침 `edited` + `original_target`)
- `dataset`: 해소되지 않은 `needs_review` 기본 제외(`--keep-unreviewed`), `uncertain_kept` 집계(`--drop-uncertain`), `--situation-key 모듈:함수` → `situation_conflicts`(+예시). `pairs` 는 off-policy 보조로 명시
- `btqa_play.suite` (`btqa-eval-suite/1`): 조건(seen·robust·unseen·train 묶음, seed·class·options·prepare·temperature·repeats), Wilson·평균 95% 구간, 다른 궤적·시작 상태 수, 경고(시작 상태 같음·탐욕 궤적 같음), 선언 안 된 옵션 조건 건너뜀. `compare` 채택 판정(`must_not_regress`·`improve_any`·`min_gain`·`tolerance`, 포화·구간 겹침 경고). CLI `suite`·`eval`·`compare`
- 기록: `StepRecorder(episode=)` → 줄마다 `episode`(seed·class·options), `write_steps`, `episode_of`(예전 기록은 run 이름 `-s<seed>`). relay `--reset-options`, `PilotRelay(options=)`
- `btqa_distill.policy.LoraPolicy` (DPO·RFT·on-policy 공통: 로드·샘플·답 구간 log π·기준 모델), `k3_kl`, `sft_nll`
- `btqa_distill.onpolicy` + CLI `pairs-onpolicy`: 학생이 간 상태(평가 steps.jsonl)에서 학생 샘플 K개를 게임 보상으로 채점 → (최고, 최저) 쌍 (margin 이상, 재생 관찰이 기록과 다르면 건너뜀, `--include-target`)
- `dpo`: `nll_coef`(chosen 토큰 평균 NLL, 기본 0.2), `val_pairs`(검증 선호 정확도·마진), epoch 마다 `adapter-ep<N>/`, 기본 lr 5e-6
- `rft`: `episodes`(스위트 `--suite --group train`), `passes`, 이점 `adv_norm`(기본 none = r − 평균), 탐욕 출력은 그룹 밖, k3 KL, `sft`·`sft_coef` NLL 섞기, `min_gap`, `follow`(best·greedy·sample), 판마다 `ckpt-<n>/`, 분기는 `btqa_play.branch.Brancher`(snapshot). 시그니처 변경: `train_rft(base, game, scratch, reward_fn, out_dir, seeds=…|episodes=…)`
- 문서 `DISTILL_SDK.md` 5장 교사 규칙(게임 무관)·6장 SFT 이후 절차·7장 게임 쪽이 줄 것
- `relay_auto(rules)` + CLI `auto --rules 모듈:함수`: 교정 모드 자동 진행 — 게임 쪽 규칙이 학생 제안을 보고 승인 / 규칙 정답으로 교정 / `{"defer"}` 교사에게. 경고·학생 오류·같은 규칙 반복에서 멈춤
- **단계 형식 `history=K` (선택, 기본 0 = 옛 형식 그대로)**: user 맨 앞에 `recent_actions` = 직전 행동 이전 K개 요약(tool·input·status·reason, 오래된 것부터). 직전 하나만 보면 A·B 번갈아·같은 행동 N번째 반복을 알 수 없다 (AI_QA r13). `actions.ActionHistory`·`history_entry`, `step_input(recent=)`, `PilotConfig(history=)`·`PilotRelay(history=)`·`run_suite(history=)`·`train_rft(history=)`·`build_onpolicy_pairs(history=)`, 옛 기록 → 새 형식 `record.with_history(records, K)` (판·step 순서, step 이 끊기면 비움) + `dataset`·`pairs` `history=`. CLI `--history` (play·relay·batch·eval·dataset·pairs·rft·pairs-onpolicy). 학습 K 와 실행 K 가 같아야 한다 — 평가 요약에 `history`
- relay 반복 표시(`repeat_no_progress`)는 진척 키 **또는** 관찰 상태(창·대화 글)가 바뀌면 진척으로 본다 (대화 넘기기 같은 버튼 반복 오탐 수정)

## 0.4.x (미배포분) — 2026-11-25

증류 후반 단계 (프로토콜·`btqa_game`·Unity 패키지 변화 없음).

- DAgger: `PilotRelay(student=...)` + `step --execute student` — 학생 행동을 실행하고 교사 라벨을 기록 (`extra.student`)
- `dataset.build_pairs` → 선호 쌍 `btqa-play-pref/1` (교사가 학생과 다르게 고른 단계: chosen=교사, rejected=학생). `build_dataset` 통계에 thought 표현 차이(`thought_variants`)를 충돌과 따로 셈
- `btqa_distill.dpo.train_dpo` (참조 = 어댑터 끈 같은 모델), `btqa_distill.rft.train_rft` (GRPO, 게임 = 채점기, scratch 게임에서 seed+prefix 재생으로 분기, reward_fn 은 게임 쪽이 준다)
- transformers 5 대응: `apply_chat_template` BatchEncoding (`train._ids`), `config.load_causal_lm` (`dtype`/`torch_dtype`)
- `train`/`dpo`/`rft`: `logits_to_keep` 로 답 구간만 logits 계산 (손실 동일, 12GB GPU 에서 VRAM 넘침 → 시스템 RAM 공유로 5배 느려지던 문제 해결). RFT 생성 `use_cache=True`
- CLI `python -m btqa_distill dpo --base --pairs --out [--beta --lr --epochs --grad-accum]`
- `build_dataset(val_runs=...)` / `dataset --val-runs`: 검증 판을 고정 (판 id 끝부분 일치) — DAgger 교정이 검증으로 빠지지 않게
- 수정: `PilotRelay` 서버가 빈 줄·JSON 아닌 요청에 죽지 않음 (오류 응답 후 계속)
- **게임 고유 처리 훅** `btqa_play.hooks.load_hook("모듈:함수")` — 증류 전처리·RFT 보상·판 준비를 게임 쪽 함수 이름으로 넘긴다 (SDK 는 그 패키지를 import 하지 않음). 게임 쪽은 `btqa_distill`(torch) 을 import 할 필요가 없어진다
- `build_dataset(correction_weight=N, prep=fn)` / `dataset --correction-weight N --prep 모듈:함수`: 교정 단계(학생 제안 ≠ 정답 또는 `edited`) 를 train 에 N 번, 샘플 전처리 (train·val). `is_correction`, 샘플 `meta.correction`, 통계 `corrections`. 행동 비교는 값이 null 인 인자를 무시
- `build_pairs(prep=fn)` + CLI `python -m btqa_play pairs ... --out --prep`
- DAgger 승인 `relay_approve(n)` / `python -m btqa_play approve [N]` — 학생 제안을 정답으로 (1~2개씩 권장)
- `btqa_distill`: `rft` CLI (`--reward`·`--prepare` 훅), `gpu-variant` (등록한 CPU 전용 모델의 GPU 변형), `export` (merge → gguf → ollama 한 번에, `--gpu-name`), `export.export_all`·`gpu_variant_modelfile`·`variant_create`

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
