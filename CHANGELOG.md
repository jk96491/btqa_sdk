# 변경 이력

## 0.1.0 — 2026-11-23

첫 판. 프로토콜 1.3.

- 계약 `docs/PROTOCOL.md` v1.3: `classes[].blocks` 추가, **학습 대상 = 블록 가중치뿐**, `run_episode.triggers` 키 = `describe.triggers.type`
- Python `btqa_game`: `GameDecl`·`ClassDecl`·`ParamDecl`·`BlockDecl`·`TriggerDecl`, `EpisodeHost`, `GameSession`, `QaGameServer`
- Python `btqa_game.conformance`: 적합성 검사 (`python -m btqa_game check --game HOST:PORT`, 표준 라이브러리만)
- Python `btqa_game.examples.minimal_arena`: 블록 이름이 다른 최소 예제 게임
- Unity `com.btqa.game` (`Btqa.Game`): `Wire`·`GameDecl`·`IEpisodeHost`·`TriggerChannel`·`QaSession`·`QaServer`
- 처음 적용한 게임: game_sim 0.5.0 (Python), QaArena 0.7.0 (Unity) — 판 결과는 SDK 도입 전과 바이트 단위 동일
