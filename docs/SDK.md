# BT-QA Game SDK 가이드

> 한 줄: **게임은 BT 를 설계하고 블록 가중치를 선언한다. 학습 플랫폼은 그 가중치만 학습해 돌려준다.**
> 계약 [`PROTOCOL.md`](PROTOCOL.md) v1.4. SDK 0.2.0.

## 1. 구성

| 부분 | 위치 | 의존성 | 하는 일 |
|---|---|---|---|
| Unity 패키지 `com.btqa.game` | `unity/com.btqa.game` (네임스페이스·어셈블리 `Btqa.Game`) | Newtonsoft.Json 만 (UnityEngine 비의존) | TCP 서버·세션, 정의서 선언(`GameDecl`), 요청·트리거 검증, 가중치만 학습 규칙 검사 |
| Python 패키지 `btqa_game` | `python/btqa_game` | 표준 라이브러리만 | Unity 판과 같은 구조 + 적합성 검사 도구 + 예제 게임 |
| 계약 | `docs/PROTOCOL.md` | — | 게임 ↔ 학습 플랫폼의 유일한 연결점 (TCP JSON Lines) |

```
┌──────────── 게임 ─────────────┐                 ┌──── 학습 플랫폼 (이 SDK 밖) ────┐
│ 게임 로직 (전투, BT)            │                 │ 최적화·RL (CMA-ES, IMPALA ...)  │
│   └ IEpisodeHost 구현 ◀─ 게임   │ ◀── TCP 1.4 ──▶ │   학습 대상 = 선언된 블록 가중치 │
│ BT-QA Game SDK                 │                 │   페르소나 = 가중치가 아닌 설정값 │
│   QaServer·QaSession·GameDecl  │                 │                                 │
└────────────────────────────────┘                 └─────────────────────────────────┘
```

학습 플랫폼 쪽은 이 SDK 를 import 하지 않는다. 프로토콜 문서만 따른다.

## 2. 핵심 규칙: 학습 대상 = 블록 가중치뿐

전략 선택 노드는 블록 점수 `S = N(상황) × w` 가 가장 큰 블록을 실행한다. N 은 게임이 짠 필요도 식, w 는 블록 가중치.

| 값 | 정하는 쪽 | SDK 가 하는 일 |
|---|---|---|
| 블록 가중치 w (`blocks[].weight`) | **학습** | describe 의 `optimize: true`, `unit: weight` 를 자동으로 채움 |
| 가중치가 아닌 BT 값 (물약 체력 등) | 정의서 default(기획 손 튜닝). 학습 플랫폼의 페르소나가 덮어쓸 수 있음 | 이 값에 `optimize: true` 를 적으면 **서버가 기동하지 않음** |
| `group: persona` 값 (예: 생존 필요도 시작 체력) | 페르소나 | 블록 가중치가 될 수 없음 |

검사 내용 (`GameDecl.validate()` / `Validate()`): 직업마다 블록 1개 이상, 가중치 파라미터가 존재하고 `group: bt`·범위 0 이상·블록마다 다른 이름, 가중치가 아닌 값에 `optimize: true` 금지.

알려진 한계: 선택이 점수 순서만 보므로 w 전체에 같은 양수를 곱해도 행동이 같다 (실효 자유도 = 블록 수 − 1). w = 0 은 '블록 끄기'라는 별도 의미가 있어 SDK 는 정규화하지 않는다.

## 3. Unity 게임에 붙이기

1. 패키지 추가 — 셋 중 하나:
   - **복사 (권장)**: `unity/com.btqa.game` 폴더를 게임 프로젝트의 `Packages/com.btqa.game` 으로 통째로 복사 (embedded 패키지, manifest 수정 없음).
     프로젝트가 SDK 위치와 무관하게 열린다. SDK 를 올릴 때 다시 복사하고, 사본은 직접 고치지 않는다.
   - 경로 참조: `Packages/manifest.json` 에 `"com.btqa.game": "file:<이 SDK 경로>/unity/com.btqa.game"`
   - git URL: `"com.btqa.game": "https://github.com/<org>/<repo>.git?path=/unity/com.btqa.game#v0.1.0"`.
   Newtonsoft(`com.unity.nuget.newtonsoft-json`)는 의존성으로 같이 들어온다.
2. `IEpisodeHost` 구현:
   ```csharp
   using Btqa.Game;
   using Newtonsoft.Json.Linq;

   public sealed class MyHost : IEpisodeHost {
       public GameDecl Declare() {
           var d = new GameDecl { Game = "my_game", GameVersion = "1.0.0", TimeLimit = 600, MaxDeaths = 3, DecisionInterval = 0.3 };
           var c = new ClassDecl { Name = "warrior" };
           c.Params.Add(new ParamDecl { Name = "w_attack", Low = 0, High = 1, Default = 0.6 });
           c.Params.Add(new ParamDecl { Name = "potion_hp", Low = 0.1, High = 0.9, Default = 0.4, Unit = "hp_ratio" });
           c.Blocks.Add(new BlockDecl("attack", "w_attack"));      // ← 학습 대상은 이렇게 선언한 가중치뿐
           d.Classes.Add(c);
           d.Scenarios.Add(new ScenarioDecl("base"));
           d.Triggers.Type = "boss_hp_below";
           d.Triggers.Supported.AddRange(new[] { 0.75, 0.5, 0.25 });
           d.Triggers.StateFeatures.AddRange(new[] { "hp_ratio", "boss_hp_ratio" });
           d.Triggers.Totals.AddRange(new[] { "elapsed", "deaths", "consumables_used", "boss_hp_remaining" });
           return d;
       }

       public JObject RunEpisode(EpisodeRequest req, TriggerChannel triggers) {
           // req.Seed 로 결정론적 판 실행 (System.Random(seed) 하나, 고정 dt 루프).
           // req.Params = 덮어쓸 값 (검증 완료, 생략한 값은 정의서 default). 경계마다:
           //   var updates = triggers.Fire(index, state, totals);   // 응답까지 게임 시간 정지, 부분 덮어쓰기
           return result;   // PROTOCOL.md 4.5
       }
   }
   ```
   블록 목록은 BT 코드에 있는 것과 같은 목록에서 만든다 (선언과 실제 BT 가 어긋나지 않게).
3. 부트 (헤드리스 빌드 `-batchmode -nographics -qaPort 7777`):
   ```csharp
   var host = new MyHost();
   host.Declare().Validate();          // 규칙 위반을 기동 시 바로 알림
   var server = new QaServer("127.0.0.1", port, () => new QaSession(host), Debug.Log);
   server.Start();                     // stdout 에 "LISTENING <port>" 를 찍어 두면 런처가 포트를 안다
   server.Serve();                     // 블로킹. 창 모드면 백그라운드 스레드에서
   ```
4. 적합성 검사: `python -m btqa_game check --game 127.0.0.1:7777` — 전 항목 PASS 면 학습 플랫폼에 붙는다.

## 4. 파이썬 게임에 붙이기

```python
from btqa_game import EpisodeHost, GameDecl, ClassDecl, ParamDecl, BlockDecl, ScenarioDecl, TriggerDecl, QaGameServer

class MyHost(EpisodeHost):
    def declaration(self):
        c = ClassDecl("warrior", params=(ParamDecl("w_attack", 0, 1, 0.6), ParamDecl("potion_hp", 0.1, 0.9, 0.4)),
                      blocks=(BlockDecl("attack", "w_attack"),))
        return GameDecl("my_game", "1.0.0", 600.0, 3, (c,), (ScenarioDecl("base"),),
                        TriggerDecl("boss_hp_below", (0.75, 0.5, 0.25), ("hp_ratio", "boss_hp_ratio"),
                                    ("elapsed", "deaths", "consumables_used", "boss_hp_remaining")))

    def run_episode(self, req, triggers):
        ...  # req.params = default + 요청값 (검증 완료). 경계마다 req 값에 triggers.fire(index, state, totals) 를 덮어씀
        return result_dict

QaGameServer(MyHost, port=7777).serve_forever()
```

전체 예제: `python/btqa_game/examples/minimal_arena.py` — `python -m btqa_game example --port 7777` 로 띄우고 `check` 로 검사.

### 4.1 판 시작 상태 (v1.4, 선택)

학습 쪽이 판 시작 상황(체력·물약)을 바꿔 가며 학습하려면 받을 수 있는 시작 상태를 선언한다. 이름은 `triggers.state_features` 중 하나.

```python
GameDecl(..., initial_state=(InitialStateDecl("hp_ratio", 0.05, 1.0, 1.0, "판 시작 체력"),
                             InitialStateDecl("potions_ratio", 0.0, 1.0, 1.0, "물약 = round(비율 × 직업 물약 수)")))
# run_episode 에서: req.initial_state 가 None 이면 지금처럼 가득 찬 상태, 아니면 {"hp_ratio": 0.7, "potions_ratio": 0.34} (default 포함)
```
Unity: `decl.InitialState.Add(new InitialStateDecl("hp_ratio", 0.05, 1.0, 1.0, "..."))`, 판에서 `request.InitialState` (null = 요청 없음). 검증(모르는 키·범위)은 SDK 가 한다.

## 5. 결정성 (가장 중요)

같은 `(class, scenario, seed, params, 트리거 응답 순서)` → 완전히 같은 result. SDK 는 이걸 대신 보장할 수 없다 — 게임이 지킨다.
판마다 seed 로 만든 난수 하나, 고정 dt 루프, 전역 난수·프레임 시간·Dictionary 순회 순서·판 사이 static 상태 의존 금지. 상세 PROTOCOL.md 6장.

## 6. 버전

| 구성 | 버전 |
|---|---|
| 프로토콜 | 1.4 (같은 주 버전 1.x 끼리 호환) |
| `com.btqa.game` / `btqa_game` | 0.2.0 (둘은 항상 같은 버전으로 올린다 — `tests/test_sdk.py` 가 검사) |

프로토콜을 바꾸면 `docs/PROTOCOL.md` → `python/btqa_game` → `unity/com.btqa.game/Runtime` → `btqa_game.conformance` 를 같이 맞추고 `CHANGELOG.md` 에 적는다.

## 7. 다음 (아직 없음)

- 결과 지표 일반화: 보상 항목을 게임이 `metrics` 로 선언
- 학습 정책 실행기 C# 판 (`policy.json` → 게임 안에서 가중치 출력)
- JSON Schema 와 언어 중립 적합성 케이스 파일
- 배포: PyPI(`btqa-game`), UPM git URL 태그
