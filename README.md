# BT-QA Game SDK

게임을 **BT 파라미터 밸런스 QA 학습 플랫폼**에 붙이는 SDK. 게임은 행동 트리(BT)를 설계하고 블록 가중치를 선언하며, 학습 플랫폼은 **그 가중치만** 학습해 돌려준다.

| | |
|---|---|
| 계약 | [`docs/PROTOCOL.md`](docs/PROTOCOL.md) — TCP JSON Lines, 밸런스 프로토콜 1.4 / [`docs/PLAY_PROTOCOL.md`](docs/PLAY_PROTOCOL.md) — 플레이 모드 0.3 |
| 가이드 | [`docs/SDK.md`](docs/SDK.md) — Unity·Python 게임에 붙이는 법 / [`docs/PLAY_SDK.md`](docs/PLAY_SDK.md) — LLM 플레이어 클라이언트 |
| Unity | [`unity/com.btqa.game`](unity/com.btqa.game) — UPM 패키지 (`Btqa.Game`, Newtonsoft.Json 만) |
| Python | [`python/btqa_game`](python/btqa_game) — 게임 쪽 (밸런스 서버), 표준 라이브러리만, Python ≥ 3.8 |
| Python | [`python/btqa_play`](python/btqa_play) — 플레이 모드 클라이언트 (LLM 도구·tool use 루프·relay·LLM 백엔드·단계 기록·평가·데이터셋), 표준 라이브러리만 |
| Python (선택) | [`python/btqa_distill`](python/btqa_distill) — 증류: LoRA 학습·GGUF·Ollama(CPU) 등록. **유일하게 외부 의존성**(torch·transformers·peft, `pip install btqa-game[distill]`) — [`docs/DISTILL_SDK.md`](docs/DISTILL_SDK.md) |
| 변경 이력 | [`CHANGELOG.md`](CHANGELOG.md) |

## 폴더

```
btqa_sdk/
├─ docs/            PROTOCOL.md·PLAY_PROTOCOL.md (계약), SDK.md·PLAY_SDK.md (가이드)
├─ unity/com.btqa.game/     Unity 패키지 (Runtime/: Wire, Declaration, Host, QaSession, QaServer)
├─ python/btqa_game/        Python 패키지 (+ conformance 적합성 검사, examples/minimal_arena 예제 게임)
├─ python/btqa_play/        플레이 모드 클라이언트 (client·tools·prompt·pilot·relay·llm/·actions·record·evaluate·dataset)
├─ python/btqa_distill/     (선택) 증류 학습·내보내기 (config·train·export)
├─ tests/           SDK 자체 테스트
├─ pyproject.toml, setup.py   pip 패키지 btqa-game
└─ CHANGELOG.md
```

## 빠른 시작

```bash
pip install -e .                                  # (setuptools < 64 이면 --no-build-isolation)
python -m btqa_game example --port 7777           # 예제 게임 서버
python -m btqa_game check --game 127.0.0.1:7777   # 적합성 검사 → 전 항목 PASS
python -m btqa_play tools --game 127.0.0.1:7800 [--spec my_game.play.json]   # (플레이 모드 게임) LLM 도구·프롬프트 미리 보기 + 선언 검사
python -m unittest discover -s tests -t .         # SDK 테스트
```

Unity: `unity/com.btqa.game` 을 게임 프로젝트 `Packages/com.btqa.game` 으로 복사 (권장, embedded 패키지 — 또는 manifest 의 `file:` 경로·git URL), 그다음 `IEpisodeHost` 구현 — [`docs/SDK.md`](docs/SDK.md) 3장.

**모델 가중치·어댑터·GGUF 는 원본이든 증류든 이 레포에 커밋하지 않는다** (`.gitignore`, `tests/test_no_model_files.py`).

## 규칙 한 줄

**학습 대상 = 전략 선택 노드의 블록 가중치 w (S = N × w) 뿐.** 게임이 `blocks` 로 선언하고, 가중치가 아닌 값에 `optimize: true` 를 두면 서버가 기동하지 않는다.
