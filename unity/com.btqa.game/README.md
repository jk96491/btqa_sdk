# com.btqa.game — BT-QA Game SDK (Unity)

BT 파라미터 밸런스 QA 학습 플랫폼(`balance_qa`)에 Unity 게임을 붙이는 패키지. 프로토콜 1.4 (`docs/PROTOCOL.md`).

- 네임스페이스 `Btqa.Game`, 어셈블리 `Btqa.Game` (UnityEngine 비의존, Newtonsoft.Json 만)
- 게임이 구현할 것: `IEpisodeHost` (`Declare()` 정의서, `RunEpisode()` 판 1개)
- SDK 가 하는 것: `QaServer`(TCP JSON Lines) · `QaSession`(hello/describe/run_episode/bye) · 요청 검증 · `TriggerChannel`(트리거 왕복) · `GameDecl.Validate()`
- **규칙: 학습 대상 = `ClassDecl.Blocks` 에 선언한 블록 가중치뿐.** describe 의 `optimize` 는 SDK 가 채운다.

| 파일 | 내용 |
|---|---|
| `Runtime/Wire.cs` | 메시지 이름·오류 코드·`ProtocolException`·`JsonLine`·`IMessageChannel` |
| `Runtime/Declaration.cs` | `ParamDecl`·`BlockDecl`·`ClassDecl`·`TriggerDecl`·`ScenarioDecl`·`GameDecl` (+ 규칙 검사, 파라미터 검증) |
| `Runtime/Host.cs` | `IEpisodeHost`·`EpisodeRequest`·`TriggerChannel` |
| `Runtime/QaSession.cs` | 연결 1개 세션 |
| `Runtime/QaServer.cs` | TCP 서버 (인스턴스 1개 = 포트 1개 = 동시 연결 1개) |

설치·사용법: SDK `docs/SDK.md` 3장. 적합성 검사: `python -m btqa_game check --game 127.0.0.1:<port>`.
Python 판은 SDK `python/btqa_game` (같은 구조·같은 규칙).

이 패키지는 SDK 레포에서만 고친다. 게임 프로젝트 쪽에서 직접 수정하지 말고 SDK 에 변경을 요청한다.
