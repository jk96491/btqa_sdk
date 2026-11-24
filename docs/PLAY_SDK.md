# BT-QA Play SDK (`btqa_play`) — LLM 플레이어 클라이언트

> 계약 [`PLAY_PROTOCOL.md`](PLAY_PROTOCOL.md) play v0.3 (+ 2026-11-24 선택 필드 4.2.1). SDK 0.4.0.
> Python ≥ 3.8, **표준 라이브러리만**. `anthropic` 백엔드만 anthropic SDK(Python ≥ 3.10)를 그 백엔드 안에서 import.

플레이 모드 = **비전투(이동·대화·퀘스트·UI)는 LLM 이, 전투는 밸런스 QA 로 학습한 BT 가** 한다.
`btqa_game` 이 게임 쪽(밸런스 프로토콜 서버)이라면, `btqa_play` 는 플레이 프로토콜의 **클라이언트 쪽**이다. 게임 쪽 플레이 서버는 게임이 직접 구현한다 (아직 SDK 없음 — 7장).

## 1. 구성

| 모듈 | 하는 일 |
|---|---|
| `client.py: PlayClient` | 프로토콜 클라이언트. `act()` 가 `action_result` 까지 읽으며 전투 메시지를 모으고, `trigger` 는 넘겨받은 응답기로 즉답. hello 0.3 → 0.2 → 0.1. `note()` = `pilot_note` |
| `spec.py` | `PlaySpec` 외부 플레이 스펙 (형식 `btqa-play-spec/1`) 읽기·검사·합치기 |
| `tools.py` | `build_tools(desc, spec)` 게임 행동 → LLM 도구 / `GameTools.execute()` 도구 호출 실행 / `validate_description()` 선언 검사 / `tool_sources()` 출처 |
| `prompt.py` | `build_system_prompt(desc, spec, extra_rules)` = SDK 공통 규칙(도구 사용 방식만) + `pilot_guide`(게임 → 스펙) + 추가 규칙, `default_goal(desc, spec)` |
| `llm/` | `LlmBackend` 확장 지점 + `AnthropicBackend` / `OpenAICompatBackend`, `load_config`·`make_backend` (`llm.json`) |
| `pilot.py: LlmPilot` | LLM tool use 루프 (행동·호출·진척 없음 상한, LLM 텍스트 → `pilot_note`) |
| `relay.py: PilotRelay` | 대화형 에이전트(Claude Code 등)가 직접 플레이어 — API 호출 없음. `step` 으로 한 행동씩 |
| `actions.py` | **단계 형식** (작은 로컬 모델·증류): [고정 system, 지금 관찰 + 직전 행동 결과] → JSON 행동 1개. `action_schema`·`parse_action`·`step_input` |
| `llm/json_action.py` | `JsonActionBackend` (`tool_mode: json`) — 출력 JSON 을 스키마로 강제 (Ollama·llama.cpp) |
| `record.py` | `StepRecorder` 단계 기록 JSONL (`btqa-play-step/1`) + `grounding_issues` |
| `__main__.py` | `python -m btqa_play tools / spec / play / relay / step` (`--spec` 외부 스펙) |

**SDK 밖 (쓰는 쪽이 만든다)**: 전투 로드아웃(`set_loadout` 항목)과 **전투 응답기**. 학습 결과 파일 형식·정책 실행(numpy 등)에 묶이기 때문이다.
응답기 = `trigger` 메시지 → `params` dict(부분 덮어쓰기) 또는 `None`(바꾸지 않음)을 돌려주는 callable. `on_combat_started(msg)` 메서드가 있으면 조우 시작 때 불린다.

## 2. SDK 에는 게임 내용이 없다 — 선언 또는 외부 스펙

SDK 는 행동 설명 문장·인자 의미·장르 개념(퀘스트·아이템 등)을 하나도 들고 있지 않다. 받은 것을 검사하고 합쳐 strict 도구·프롬프트로 바꿀 뿐이다.

```
행동마다 (설명·스키마 따로), pilot_guide 키마다, progress_keys:
   1. 게임 선언  play_description.actions[].description/schema, pilot_guide, progress_keys   (PLAY_PROTOCOL 4.2.1)
   2. 외부 스펙  쓰는 쪽이 넘기는 JSON 파일 (형식 아래)                                    ← 게임이 아직 선언하지 않을 때
   3. 일반형     args 이름마다 nullable 문자열, 설명 = 이름                               ← 검사가 경고
+ 메타 도구 inspect_ui, finish (스키마 = SDK, 설명은 스펙 meta_tools 로 바꿀 수 있음)
```

### 2.1 외부 플레이 스펙 형식 `btqa-play-spec/1`

```json
{
  "format": "btqa-play-spec/1",
  "actions": {
    "move_to": {"description": "...", "schema": {"type": "object", "properties": {"target": {"type": "string"}, "pos": {"type": "array", "items": {"type": "number"}}}, "required": []}},
    "interact": {"description": "...", "schema": {"type": "object", "properties": {"target": {"type": "string"}, "verb": {"type": "string", "enum": ["talk", "open"]}}, "required": ["target", "verb"]}}
  },
  "meta_tools": {"finish": {"description": "..."}},
  "pilot_guide": {"system_prompt": "관찰 읽는 법·큰 규칙 (문단)", "about": "한 줄 소개", "rules": ["요령"], "default_goal": "목표"},
  "progress_keys": ["quests", "player.zone"]
}
```

| 키 | 규칙 |
|---|---|
| `format` | `"btqa-play-spec/1"` 필수 |
| `actions.<이름>` | `{description?, schema?}` — 게임 선언 `actions[]` 항목과 같은 의미. 스키마 규칙은 PLAY_PROTOCOL 4.2.1 (`required` 생략 = 모두 필수, 필수가 아니면 null 허용으로 바뀜). 게임이 지원하지 않는 이름은 무시(검사가 알림) |
| `meta_tools.<inspect_ui\|finish>` | `{description}` 만 |
| `pilot_guide` | `system_prompt`·`about`·`rules`·`default_goal` — 게임 선언과 키마다 합쳐진다 (게임이 우선) |
| `progress_keys` | 진척 판단 관찰 경로 (점 구분). 없으면 관찰 전체 − 매번 바뀌는 필드 |
| `_` 로 시작하는 키 | 주석 (무시) |

- 검사: `python -m btqa_play spec <파일>` (게임 없이) / `python -m btqa_play tools --game HOST:PORT --spec <파일>` (출처 표 `game`·`spec`·`generic`, 선언 검사, 완성된 프롬프트, `--json` = 도구 전체).
- 코드: `PlaySpec.load(path)` / `PlaySpec.from_dict(d)`, `LlmPilot(..., spec=...)`, `PilotRelay(..., spec=...)` (경로·dict·PlaySpec 모두 받음).
- 시스템 프롬프트 = SDK 공통 규칙(도구 사용 방식만) + `pilot_guide.system_prompt` + "<game> 안내:" (`about` + `rules` + 쓰는 쪽 `extra_rules`).

## 3. 쓰는 법

```python
from btqa_play import LlmPilot, PilotConfig, PlayClient
from btqa_play.llm import load_config, make_backend

backend = make_backend(load_config(None, {"llm": "openai", "model": "qwen2.5:14b"}))
with PlayClient.connect("127.0.0.1:7800") as c:
    c.hello(); desc = c.describe()
    c.set_loadout(my_entries, persona="stable")        # 학습 결과 → 항목 (쓰는 쪽)
    obs = c.reset(seed=1, cls="warrior")
    run = LlmPilot(c, responder=my_responder, config=PilotConfig(max_actions=40), backend=backend,
                   spec="my_game.play.json").run(obs)      # 게임이 아직 선언하지 않은 설명·안내
    log = c.get_log()                                   # 재현 = 같은 seed + 같은 행동열
```

대화형 에이전트가 플레이어일 때 (API 없음):

```python
class MyRelay(PilotRelay):
    def prepare(self, client, desc):          # reset 전에 한 번
        client.set_loadout(my_entries)
        return my_responder
    def run_meta(self):                       # run.json 에 덧붙일 것
        return {"loadout": ["reports/..."]}

MyRelay("127.0.0.1:7800", cls="warrior").serve()
```
```bash
python -m btqa_play step intro                                         # 처음 관찰·도구·규칙·목표
python -m btqa_play step move_to '{"target":"npc_elder","pos":null}' --think "촌장에게 먼저" --kind plan
python -m btqa_play step finish '{"success":true,"summary":"...","blocked":null}'
```
로드아웃 없이 게임 default BT 로 바로 해 보려면 `python -m btqa_play relay --game ...` / `python -m btqa_play play --game ... --llm openai --model ...`.

## 4. LLM 고르기 (`llm.json`)

SDK 는 특정 LLM·키를 들고 있지 않다. 예시 `btqa_play/llm.example.json` 을 작업 폴더에 `llm.json` 으로 복사 (git 에 올리지 않는다). CLI 인자가 파일보다 우선.

| 키 | 뜻 |
|---|---|
| `llm` | `openai`(OpenAI 호환: Ollama·LM Studio·llama.cpp·vLLM·호환 클라우드) / `anthropic`(Claude API). 필수 |
| `model` | openai 필수, anthropic 기본 `claude-opus-5-5` |
| `base_url` | openai 서버 (기본 `http://localhost:11434/v1` = Ollama) |
| `api_key_env` | 키가 든 **환경변수 이름**. 키 값을 파일에 쓰면 거부 |
| `effort` | anthropic 만 |
| `max_tokens`, `temperature`, `timeout`, `extra_body` | 선택 |

새 백엔드: `llm/base.py: LlmBackend` 상속 (`start / complete / add_tool_results / add_user`) → `llm/__init__.py: make_backend` 에 등록. 도구는 중립 형식 `{name, description, input_schema, strict}` 로 들어온다.

## 4.1 작은 로컬 모델 (CPU)

```bash
ollama pull qwen2.5:1.5b
python -m btqa_play play --game 127.0.0.1:7800 --llm openai --model qwen2.5:1.5b --tool-mode json --stateless [--record steps.jsonl]
```

- `--stateless`: 단계마다 새 대화 = [system, 지금 관찰 + 직전 행동 결과]. system 은 판 내내 같은 글자라 llama.cpp/Ollama 가 앞부분 계산을 재사용한다 → CPU 에서도 단계마다 새로 읽는 것은 관찰뿐.
- `--tool-mode json`: 출력이 `{"thought","tool","input"}` JSON 하나로 강제된다 (서버가 `response_format: json_schema` 를 지원해야 함 — Ollama 0.5+). 형식이 깨지면 `PilotRun.invalid_outputs` 에 세고 다음 단계 입력의 `last_action.error` 로 알린다.
- 입력을 더 줄이려면 스펙·게임 선언의 `observation_drop` (2.1).

## 4.2 증류 데이터 (교사 → 작은 모델)

- 교사 = 대화형 에이전트: `python -m btqa_play relay --game ... --record teacher.jsonl` → `step intro` 의 `system`·`step_input` 을 읽고, 매 행동 응답의 `step_input`(학생이 받을 입력 그대로)만 보고 다음 행동을 정한다. `--think` = 학습 목표의 thought.
- 한 줄 = 학습 샘플 하나: `messages`(system·user) + `target_text`(assistant). `grounding` 이 비어 있지 않은 줄은 학생 입력에 근거가 없는 판단이므로 빼거나 검토한다.
- 모델 기록 (`play --record`) 도 같은 형식 — 평가·교정(DAgger) 데이터.
- 평가 `python -m btqa_play batch` (evaluate.run_batch), 학습 데이터 `python -m btqa_play dataset` (dataset.build_dataset), 학습·내보내기는 선택 패키지 `btqa_distill` — [`DISTILL_SDK.md`](DISTILL_SDK.md).

## 5. 결과

`LlmPilot.run()` → `PilotRun` (끝난 이유 `finish`/`max_actions`/`max_calls`/`no_progress`/`game_error: ...`, finish 요약, 전투 요약, 토큰, 대화 기록). `PilotRelay` 는 `<out>/relay-<시각>/run.json`·`play_log.json`.
전투 요약(`Combat.summary()`)의 `out_of_training_dist` = 물약을 cap 보다 적게 가져간 전투 (학습 분포 밖, 프로토콜 4.7).

## 6. 버전

| | |
|---|---|
| `btqa_play` / `btqa_game` / `com.btqa.game` | 0.4.0 (레포 하나, 같은 버전 — `tests/test_play.py`·`test_sdk.py` 가 검사) |
| 플레이 프로토콜 | 0.3 (`PlayClient` 는 0.2·0.1 게임에도 접속, 메모는 0.3 만) |

## 7. 다음 (아직 없음)

- 게임 쪽 플레이 서버 SDK (Unity C#: 관찰 추출·행동 실행·조우 인계) — 지금은 게임이 직접 구현 (QaArena)
- MCP 서버 래퍼 (`GameTools` 를 MCP 도구로 노출 → 대화형 에이전트가 relay CLI 없이)
- 플레이 프로토콜 적합성 검사 공용판 (지금은 쓰는 쪽 `check-play` 가 게임별 시나리오와 함께 검사)
