# 변경 이력

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
