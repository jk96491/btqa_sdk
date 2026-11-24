# 변경 이력

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
