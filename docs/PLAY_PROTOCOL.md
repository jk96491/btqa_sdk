# 플레이 모드 프로토콜 v0.3 — LLM 비전투 플레이 + 학습된 BT 전투

> 상태: **v0.3** (2026-11-23) = v0.2 + 플레이어 생각 메모 `pilot_note`(4.10). 요청 `unity_game/PLAY_PROTOCOL_0.3_REQUEST.md`.
> **원본 = BT-QA SDK 레포 `docs/PLAY_PROTOCOL.md`** (SDK 0.3.0, 2026-11-24). 클라이언트 쪽 구현 = SDK `btqa_play` (5장). 게임 쪽(서버)은 게임이 직접 구현한다.
> 2026-11-24 추가 (선택 필드, 버전 0.3 유지 — 7장): 행동 선언 `actions[].description`·`actions[].schema`, 게임 안내 `pilot_guide`, 진척 기준 `progress_keys` (4.2.1). 선언하지 않으면 클라이언트가 같은 형식의 외부 스펙 파일로 채운다 (SDK 에는 게임 내용이 없다).
> v0.2 (2026-11-23): v0.1 초안을 Unity QaArena **0.6.2** 구현(`unity_game/PLAY_MODE_REPORT.md` 2장)에 맞춰 확정했다.
> 0.6.2 가 아직 따르지 않는 항목은 **[0.6.2 미반영]** 으로 표시하고 `unity_game/PLAY_PROTOCOL_0.2_FOLLOWUP.md` 로 요청했다.
> 밸런스 QA 계약(`PROTOCOL.md` v1.2)과는 **별개의 계약**이다. Unity 작업 목록 `unity_game/PLAY_MODE_REQUEST.md`(반영됨).

## 0. 목적

- 튜토리얼·퀘스트·이동·대화·아이템 같은 **비전투**는 LLM 이 직접 플레이한다. 레이턴시(수 초)는 허용한다.
- **전투**는 밸런스 QA 로 최적화한 BT(파라미터 = `reports/<run>/params.json`, 선택적으로 IMPALA 트리거 정책 `policy.json`)가 맡는다.
- LLM 은 화면(비전)이 아니라 **Unity 런타임 계층 정보에서 뽑은 구조화된 관찰**을 읽는다.

### 0.1 두 모드 (사용자 결정 2026-11-22)

| | 학습 (밸런스 QA) | 플레이 (학습 이후) |
|---|---|---|
| 실행 | `-qaPort` (지금 그대로) | `-qaPlay -qaPort` |
| 계약 | `PROTOCOL.md` v1.2 | 이 문서 |
| 시작 지점 | **전투로 바로** — `run_episode` 한 번 = 보스전 한 판. 마을·퀘스트·이동 없음 | **게임 처음부터** — `reset` 은 항상 새 게임(튜토리얼 시작). 중간 지점에서 시작하는 옵션 없음 |
| 누가 조종 | BT (학습 쪽은 파라미터만 보냄) | 비전투 = LLM, 전투 = 학습된 BT |
| 전투 로직 | `CombatSim` + `BTAgent` | **같은** `CombatSim` + `BTAgent` (파라미터 = 학습 결과) |

- 필드에서 만나는 보스도 학습 때는 밸런스 모드 시나리오로 **전투만 바로** 돌려서 학습한다 (예: `optimize --scenario forest_wolf_king`).
- 플레이 모드는 학습 데이터를 만들지 않는다. 학습 결과를 실제 진행 흐름 속에서 검증하는 용도다.

## 1. 원칙

| 원칙 | 내용 |
|---|---|
| 게임이 선언, 클라이언트는 그것만 본다 | 관찰 대상·가능한 행동은 게임이 `QaTag` 어노테이션과 uGUI 자동 추출로 공개한다. 클라이언트는 씬 구조·스크립트를 모른다 |
| 플레이어가 지각 가능한 것만 | 기본 관찰은 화면에 보이는 것만 (카메라 시야, UI 가림 검사). 숨은 정보는 디버그 옵션으로만 (4.2) |
| 행동은 고수준 | `move_to(id)`, `interact(id)`, `ui_click(id)` 등. 길찾기·입력 시뮬레이션은 게임이 한다 |
| 클라이언트가 생각하는 동안 세계 정지 | 게임은 요청을 처리할 때만 시간이 흐른다 (턴제). LLM 레이턴시가 플레이 결과에 영향을 주지 않는다 |
| 전투 진입은 게임이 판단 | 어그로 반경 진입 등은 게임이 매 틱 감지하고 **즉시 BT 로 전환**. LLM 판단을 기다리지 않는다 |
| 전투 로직 = `CombatSim` | 필드 전투도 학습에 쓴 것과 같은 `CombatSim`·`BTAgent`·같은 인자로 돌리고 화면만 다르게 보인다 (sim-to-real 격차 0) |

## 2. 전송

- `PROTOCOL.md` 2장과 같다: TCP, UTF-8(BOM 없음) JSON 1개 + `\n`, 게임 = 서버, 인스턴스당 연결 1개, `error` 후 연결 유지.
- 실행: `QaArena.exe -qaPlay -qaPort 7800 [-qaPlaySeed 1] [-qaPlaySpeed 1.0] [-qaPlayOmniscient]` (창 모드). 헤드리스는 `-batchmode -nographics` 추가.
  - `-qaPlay` 가 없으면 기존 밸런스 QA 서버(v1.2) 그대로. 한 프로세스는 둘 중 하나만 말한다 (밸런스 인스턴스는 `reset`·`play_describe` 에 `unknown_type`).
  - `-qaPlaySpeed` 는 관람 배속일 뿐이다. 결과는 고정 dt·틱 수로만 정해져 **창 모드 ↔ 헤드리스 결과·로그가 같다**.
- **게임 → 클라이언트 메시지는 클라이언트 요청이 처리 중일 때만** 온다. 대기 중에는 아무것도 보내지 않고 세계는 정지해 있다.
- **세션 = 연결**. 새 연결은 `not_reset` 상태에서 시작하고 로드아웃도 비어 있다.

## 3. 메시지 흐름

```
클라이언트(play_agent: LLM 루프 + 전투 트리거 응답기)          게임 (-qaPlay)
  │ ── hello {protocol:"play"} ──────────────────────────────▶ │
  │ ◀─────────────────────────────────────────── hello_ok ─── │
  │ ── play_describe ───────────────────────────────────────▶ │
  │ ◀──────────────────────────────────────── play_description │  (행동·직업·시나리오·조우·퀘스트)
  │ ── set_loadout ((class, scenario)별 BT params) ─────────▶ │
  │ ◀──────────────────────────────────────────── loadout_ok ─ │
  │ ── reset {seed, class}  (항상 새 게임) ─────────────────▶ │
  │ ◀────────────────────────────────────────── observation ── │
  │ ── act {move_to npc_elder} ─────────────────────────────▶ │  세계 시간 흐름
  │ ◀──────────────────── action_result {done, observation} ── │  세계 정지
  │ ── act {move_to [0,0,32]} ──────────────────────────────▶ │
  │ ◀────────────────────── combat_started {encounter, ...} ── │  BT 가 즉시 조종
  │ ◀──────────────────── trigger {index 0, ...} (요청 시) ─── │  전투 시간 정지
  │ ── trigger_response {params} ───────────────────────────▶ │  (LLM 아님, 정책이 즉답)
  │                  ...                                       │
  │ ◀───────────────────────── combat_ended {result, ...} ──── │
  │ ◀── action_result {interrupted, by:"combat", observation} ─ │  (engage 였다면 done)
  │ ── bye ─────────────────────────────────────────────────▶ │
```

## 4. 메시지 명세

### 4.1 `hello` → `hello_ok`

```json
{"type":"hello","protocol":"play","protocol_version":"0.2","client":"play_agent"}
{"type":"hello_ok","protocol":"play","protocol_version":"0.2","game":"qa_arena","game_version":"0.6.2"}
```
- `protocol` 이 `"play"` 가 아니면(밸런스 클라이언트가 접속) `error(protocol_version)`.
- **0.x 동안은 주.부 버전이 모두 같아야 한다** (7장). 다르면 `error(protocol_version)`.
  - [0.6.2 미반영] 0.6.2 는 `"0.1"` 만 받는다. 내용은 이 문서와 같고 아래 미반영 항목만 다르다. `play_agent` 는 `"0.2"` 가 거절되면 `"0.1"` 로 다시 시도하고 그 사실을 기록한다.

### 4.2 `play_describe` → `play_description`

```json
{
  "type": "play_description",
  "protocol_version": "0.2", "game": "qa_arena", "game_version": "0.6.2",
  "classes": ["warrior","mage","archer"],
  "bosses": [{"name":"default","display_name":"default"}, {"name":"forest_wolf_king","display_name":"forest_wolf_king"}],
  "scenarios": [{"name":"base","boss":"default"}, {"name":"patch_boss_fast","boss":"default"}, {"name":"forest_wolf_king","boss":"forest_wolf_king"}],
  "actions": [
    {"name":"move_to","args":{"target":"entity_id | null","pos":"[x,y,z] | null"}},
    {"name":"interact","args":{"target":"entity_id","verb":"talk|pickup|open|enter"}},
    {"name":"ui_click","args":{"ui_id":"string"}},
    {"name":"use_item","args":{"item_id":"string"}},
    {"name":"engage","args":{"target":"entity_id"}},
    {"name":"wait","args":{"seconds":"number | null","until":"event_kind | null"}}
  ],
  "limits": {"max_action_seconds": 60.0, "interact_range": 2.5},
  "observation_modes": ["semantic","hierarchy"],
  "debug": {"omniscient": false},
  "combat": {
    "trigger_state_features": ["hp_ratio","boss_hp_ratio","potions_ratio","resource_ratio","burst_ready","time_ratio","deaths_ratio"],
    "trigger_totals": ["elapsed","deaths","consumables_used","boss_hp_remaining"],
    "encounter_start_state": "barrier_inventory_potions",
    "encounters": [{"enemy":"wolf_king","zone":"forest","scenario":"forest_wolf_king","boss":"forest_wolf_king"}]
  },
  "quests": [{"id":"q_tut_01","title":"촌장을 찾아가자"}],
  "pilot_notes": {"max_chars": 200, "kinds": ["thought","plan","finding"]},
  "stage": "P3"
}
```

| 필드 | 의미 |
|---|---|
| `scenarios` | 밸런스 모드 `describe.scenarios` 와 **같은 이름** + 그 시나리오의 보스. 필드 조우는 시나리오 하나에 대응하고 **로드아웃 키 = (class, scenario)** |
| `actions` | 게임이 **실제로 지원하는 행동만**. 클라이언트는 이 목록으로 LLM 도구를 만든다. 목록에 없는 행동 → `error(unknown_action)` (0.6.2: `ui_input` 없음) |
| `combat.trigger_*` | 전투 트리거의 키. 밸런스 프로토콜 `describe` 의 `state_features`·totals 와 **같은 이름·순서**여야 학습한 정책을 그대로 쓴다 |
| `combat.encounter_start_state` | (v0.3) `"barrier_inventory_potions"`: 체력·자원 = full(보스 결계), **물약 수 = 인벤토리** (4.7). v0.2 이하 `"full"`: 물약까지 학습 판 시작과 같음 |
| `combat.encounters` | 필드에서 만날 수 있는 적 엔티티 → 시나리오. 클라이언트가 **필요한 로드아웃 목록**을 미리 안다 |
| `quests` | 게임의 퀘스트 id·제목 (진척은 관찰 `quests`) |
| `debug.omniscient` | `-qaPlayOmniscient` 로 켰을 때만 true |
| `stage` | 구현 단계 표시(정보용). 클라이언트는 해석하지 않는다 |
| `pilot_notes` | (v0.3) 게임이 `pilot_note`(4.10)를 화면에 표시할 수 있을 때만 넣는다. 없으면 클라이언트는 보내지 않는다 |
| `actions[].description`, `actions[].schema` | (선택, 2026-11-24) LLM 도구 설명·인자 JSON Schema — 4.2.1 |
| `pilot_guide` | (선택, 2026-11-24) LLM 플레이어에게 줄 게임 안내 — 4.2.1 |
| `progress_keys` | (선택, 2026-11-24) LLM 진척 판단에 볼 관찰 경로 — 4.2.1 |
| `observation_drop` | (선택, 2026-11-24) LLM 에게 보이기 전에 뺄 관찰 경로 — 4.2.1 |
| `reset_options` | (선택, 2026-11-26) `reset.options` 로 고를 수 있는 시작 조건 선언 — 4.4.1 |
| `capabilities.snapshot` | (선택, 2026-11-26) `snapshot`/`restore` 지원 — 4.11 |

#### 4.2.1 LLM 도구·안내 선언 (선택 필드)

클라이언트(SDK `btqa_play`)는 `actions` 로 LLM 도구를 만든다. **SDK 는 행동 설명·인자 의미·게임 안내를 하나도 들고 있지 않다.** 출처는 키마다 다음 순서다:

1. **게임 선언** — 아래 필드 (이 게임의 정본)
2. **외부 플레이 스펙** — 쓰는 쪽이 넘기는 같은 형식의 JSON 파일 (`btqa-play-spec/1`, `PLAY_SDK.md` 2장). 게임이 아직 선언하지 않을 때 쓴다
3. **일반형** — `args` 이름마다 nullable 문자열, 설명 = 이름. 검사(`python -m btqa_play tools`)가 경고한다

→ 게임은 이 필드를 선언하는 것이 원칙이다. 선언하면 외부 스펙 없이 어떤 클라이언트든 같은 도구를 쓴다.

```json
"actions": [
  {"name": "move_to", "args": {"target": "entity_id | null", "pos": "[x,y,z] | null"},
   "description": "엔티티(target) 또는 좌표(pos) 로 걸어간다. 둘 중 하나만. 이동 중 어그로 범위에 들면 전투가 자동 시작되고 학습된 BT 가 싸운다.",
   "schema": {"type": "object",
              "properties": {"target": {"type": "string", "description": "entities[].id"},
                             "pos": {"type": "array", "items": {"type": "number"}, "description": "[x,y,z]"}},
              "required": []}},
  {"name": "interact", "args": {"target": "entity_id", "verb": "talk|pickup|open|enter"},
   "description": "엔티티와 상호작용. 범위 밖이면 out_of_range — 먼저 move_to.",
   "schema": {"type": "object",
              "properties": {"target": {"type": "string"}, "verb": {"type": "string", "enum": ["talk", "pickup", "open", "enter"]}},
              "required": ["target", "verb"]}}
],
"pilot_guide": {
  "about": "마을과 숲으로 된 튜토리얼 RPG. 촌장에게서 퀘스트를 받아 숲의 늑대왕을 잡는다.",
  "rules": ["경비병은 마을 남동쪽 문 앞에 있다", "상점 구매 전 골드를 확인한다"],
  "default_goal": "튜토리얼 퀘스트를 모두 끝까지 완료하라 (진행 중인 퀘스트가 없어질 때까지).",
  "system_prompt": "관찰 필드를 읽는 법, 이 게임에서 지켜야 할 큰 규칙 (여러 줄 문단)"
},
"progress_keys": ["quests", "player.zone", "inventory", "player.gold", "player.quickslots"]
```

| 필드 | 규칙 |
|---|---|
| `description` | 비어 있지 않은 문자열. LLM 이 읽는 도구 설명 — 언제 쓰는지, 실패 `reason` 이 무엇을 뜻하는지 |
| `schema` | JSON Schema `{"type":"object","properties":{...},"required":[...]}`. 속성 `type` 은 JSON Schema 타입(`string`·`number`·`integer`·`boolean`·`array`·`object`·`null`) 또는 `anyOf`/`enum`. **속성 이름 = `args` 키** (같은 집합). `required` 생략 = 모두 필수. 필수가 아닌 인자는 클라이언트가 null 허용으로 바꾸고, null 인자는 `act.args` 에서 뺀다 |
| 이름 | `inspect_ui`·`finish` 는 클라이언트 메타 도구라 행동 이름으로 쓰지 않는다 |
| `pilot_guide.about` | 게임 한 줄 소개 (문자열) |
| `pilot_guide.system_prompt` | 관찰 필드를 읽는 법·이 게임의 큰 규칙 (문단). SDK 공통 규칙(도구 사용 방식만: 한 번에 하나, reason 읽기, 전투는 BT, finish) 바로 뒤에 붙는다 |
| `pilot_guide.rules` | 이 게임만의 플레이 요령 (문자열 목록). `about` 과 함께 "<game> 안내:" 블록이 된다 |
| `pilot_guide.default_goal` | 목표를 따로 주지 않았을 때 LLM 목표 (문자열) |
| `progress_keys` | 진척 판단에 볼 관찰 경로 목록 (점 구분, 예 `player.zone`). 이 값들이 N 행동 동안 안 바뀌면 LLM 루프가 `no_progress` 로 멈춘다. 없으면 관찰 전체에서 매 행동 바뀌는 필드(`seq`·`game_time`·`events`·`entities`·`player.pos`)를 뺀 것 |
| `observation_drop` | LLM 에게 보이기 전에 관찰에서 뺄 경로 목록 (입력 길이 — 작은 로컬 모델 CPU 추론). `a.b` = 키, 목록이면 원소마다, 마지막 조각 `name[k=v]` = 목록에서 k==v 인 원소만 (예 `"screen.ui[role=text]"`). 결정에 필요한 것은 빼지 않는다 |
| 그 밖의 키 | `pilot_guide` 에 다른 키 → 클라이언트 검사(`python -m btqa_play tools`)가 문제로 보고한다 |

- 세션 동안 바꾸지 않는다 (LLM 프롬프트 캐시). 플레이어에게 보이지 않는 정보(숨은 좌표 등)를 `rules` 에 넣지 않는다 — 원칙 "플레이어가 지각 가능한 것만" (1장). 사람 플레이어가 튜토리얼·도움말에서 알 수 있는 정도까지.
- 확인: `python -m btqa_play tools --game HOST:PORT` → 행동마다 설명/스키마 출처(`game`·`sdk`·`generic`), 선언 검사, 완성된 시스템 프롬프트.

### 4.3 `set_loadout` → `loadout_ok`

전투에서 BT 가 쓸 파라미터. 학습 결과 `params.json` 의 `source.params` 를 그대로 옮긴다.

```json
{"type":"set_loadout","persona":"stable",
 "entries":[
   {"class":"warrior","scenario":"forest_wolf_king","boss":"forest_wolf_king",
    "params":{"w_burst":1.0,"w_dps":0.858,"w_survival":0.730,"potion_hp":0.839,"resource_reserve":0.279,"min_commit":0.213,
              "survival_hp_threshold":0.8,"dps_hp_floor":0.7,"charge_distance":7.31},
    "triggers":{"boss_hp_below":[0.75,0.5,0.25]},
    "source":"reports/20261123-001010 (cma)"}
 ]}
{"type":"loadout_ok","entries":1}
```

- 항목 키 = **(class, scenario)** (밸런스 프로토콜에서 판을 정하는 것). `boss` 는 생략 가능한 확인용: 그 시나리오의 보스와 다르면 `error(boss_mismatch)`.
  - [0.6.2 미반영] 0.6.2 는 이 경우 `unknown_boss` 를 보낸다. `play_agent` 는 둘 다 같은 오류로 다룬다.
- 검증은 `PROTOCOL.md` 4.3 `run_episode.params` 와 같다: `unknown_class`, `unknown_scenario`, `unknown_param`, `param_out_of_range`, `boss_mismatch`. 실패하면 로드아웃 전체를 바꾸지 않는다.
- `set_loadout` 은 **로드아웃 전체를 교체**한다 (누적 아님). 연결이 끊기면 사라진다.
- `triggers` 가 있으면 그 조우에서 `trigger` 를 보낸다 (4.7). 없으면 고정 파라미터로 끝까지 싸운다.
- 조우 시나리오에 맞는 항목이 없으면 정의서 default(손 튜닝값)로 싸우고 `combat_started.loadout.matched=false`.
- `persona`·`source` 는 기록용 문자열. 게임은 해석하지 않고 `combat_started`·로그에 그대로 싣는다.

### 4.4 `reset` → `observation`

```json
{"type":"reset","seed":1,"class":"warrior"}
```
- `reset` = **항상 새 게임** (튜토리얼 첫 장면, 퀘스트·인벤토리·골드 초기 상태). 중간 지점에서 시작하는 옵션은 없다.
  - 게임 안의 체크포인트(`cp_forest`)는 전투 패배 시 **부활 위치**일 뿐, 시작 지점으로 고를 수 없다.
- 인자는 **`seed`·`class`** (+ 게임이 `reset_options` 를 선언했으면 `options`, 4.4.1). 다른 키 → `error(bad_request)`. `seed` 생략 = `-qaPlaySeed`(기본 1), `class` 생략 = `warrior`. 모르는 class → `unknown_class`.
- `seed`: 세계 난수와 조우 seed 의 근원 (4.7 조우 seed). **seed 가 바꾸는 것(배치·수량·순서 등)은 게임이 정하고 문서에 적는다** — 평가에서 seed N개가 서로 다른 판이어야 표본 N개다 (4.4.1 참고).
- 응답은 첫 `observation` (`events` 에 `new_game`).
- (2026-11-26) `options` — 게임이 `reset_options` 를 선언했을 때만 (4.4.1). 선언하지 않은 게임에 보내면 `bad_request` (위 규칙 그대로).

#### 4.4.1 `reset.options` — 선언한 시작 조건 (선택, 2026-11-26)

평가·학습이 **안 본 시작 조건**(일반화)과 **흔들린 조건**(강건성)을 고를 수 있게, 게임이 고를 수 있는 시작 조건을 선언한다.
SDK 는 이름·범위만 확인하고 뜻은 모른다 (밸런스 프로토콜 1.4 `initial_state` 와 같은 방식).

```json
"reset_options": [
  {"name": "start_gold", "type": "int", "min": 0, "max": 200, "default": 20, "description": "시작 골드"},
  {"name": "layout",     "type": "enum", "values": ["a", "b", "c"], "default": "a", "description": "채집물 배치 변형"}
]
{"type": "reset", "seed": 101, "class": "archer", "options": {"start_gold": 0, "layout": "c"}}
```
| 규칙 | |
|---|---|
| 형식 | `name`(필수, 영문 소문자·`_`), `type` = `int`·`number`·`bool`·`enum`, `min`·`max`(수), `values`(enum), `default`, `description` |
| 검사 | 모르는 이름·범위 밖·형식 오류 → `error(bad_request)` (reset 전체 무효, 이전 상태 유지). 빠진 키 = `default` |
| 결정성 | 같은 `seed` + 같은 `options` + 같은 행동열 → 같은 로그 (4.8). `play_log` 의 `reset` 항목에 `options` 를 남긴다 |
| 관찰 | 옵션 값 자체를 관찰에 몰래 넣지 않는다 — 플레이어가 지각할 수 있는 결과(골드 표시, 채집물 위치)만 보인다 (1장) |
| 클라이언트 | SDK `PlayClient.reset(options=...)` 는 선언하지 않은 키를 보내기 전에 거부한다. 평가 스위트(`btqa-eval-suite/1`)의 조건 `options` 는 게임이 선언하지 않았으면 건너뛰고 이유를 남긴다 |

### 4.5 `observe` → `observation`

```json
{"type":"observe","mode":"semantic"}
{"type":"observe","mode":"hierarchy","root":"ui.quests","max_depth":3}
```

**semantic (기본)** — 게임이 정리한 상태:

```json
{
  "type":"observation","seq":12,"game_time":134.6,
  "player":{"class":"warrior","zone":"village","pos":[12.0,0.0,-3.5],"hp_ratio":1.0,"level":1,"gold":20,
            "state":"idle","quickslots":["potion_small",null]},
  "quests":[{"id":"q_tut_02","title":"숲의 약초","state":"active",
             "objectives":[{"id":"o1","text":"약초 3개 채집","progress":[1,3],"done":false,"hint_targets":["herb_01","herb_02"]}]}],
  "screen":{"modal":"dialog",
            "ui":[{"id":"ui.dialog.text","role":"text","text":"약초를 구해다 주겠나?"},
                  {"id":"ui.dialog.choice.0","role":"button","text":"맡겨 주세요","interactable":true},
                  {"id":"ui.dialog.choice.1","role":"button","text":"나중에","interactable":true}]},
  "entities":[{"id":"npc_elder","role":"npc","label":"촌장","pos":[14.0,0.0,-2.0],"distance":2.5,
               "actions":["talk"],"marker":"quest_available"},
              {"id":"herb_01","role":"item","label":"약초","distance":31.2,"actions":["pickup"],"marker":"objective"}],
  "inventory":[{"id":"potion_small","label":"작은 물약","count":3,"actions":["use"]}],
  "events":[[130.1,"quest_accepted",{"quest":"q_tut_02"}]],
  "hidden_included":false
}
```

| 필드 | 규칙 |
|---|---|
| `player.state` | `idle` \| `moving` \| `dialog` |
| `player.quickslots` | 길이 2, 빈 칸은 `null` |
| `quests` | **진행 중인 것만**. `state` = `active` \| `ready`(목표 완료, 반납만 남음). 목표 `{id, text, progress:[n,N], done, hint_targets}`. 완료된 퀘스트는 빠지고 `events` 의 `quest_completed` 로 알 수 있다 |
| `entities` | `QaTag` 가 붙은 월드 오브젝트 중 **지각 가능한 것** (카메라 시야 안, 또는 `alwaysKnown` 예: `portal_forest`). 거리순, 최대 32개 |
| 엔티티 `role` | `npc` \| `item` \| `portal` \| `container` \| `enemy` \| `area` |
| 엔티티 `marker` | `quest_available` \| `quest_turn_in` \| `objective` (없으면 생략) |
| `screen.ui` | 활성 Canvas 의 uGUI 요소 + `QaTag(ui)` 보강. `interactable` = `Selectable.IsInteractable()` **그리고** 요소 중심 EventSystem 레이캐스트가 자기 자신에 닿음 (가려진 버튼은 false) |
| `screen.modal` | 입력을 막는 창: `null` \| `dialog` \| `quests` \| `inventory` \| `shop` \| `tip`(창 위에 뜨는 안내 팝업) |
| `events` | 직전 관찰 이후 발생한 사건 `[시간, 종류, 내용]` |
| `hidden` | omniscient 일 때만: 시야 밖·닫힌 창 요소에 `"hidden": true`, `hidden_included: true` |
| id | **안정적**: 같은 오브젝트는 세션 내내 같은 id. UI id 목록은 `unity_game/README.md` "플레이 모드" 절 |
| 텍스트 | 플레이어에게 보이는 문자열 그대로 (리치텍스트 태그 제거) |

**hierarchy** — 계층 구조를 그대로 보는 탐색·디버그용:

```json
{"type":"observation","seq":13,"mode":"hierarchy","root":"ui.quests",
 "tree":{"id":"ui.quests","name":"QuestWindow","active":true,"components":["RectTransform","CanvasGroup"],
         "children":[{"id":"ui.quests.close","name":"Close","components":["Button","Image"],"text":"닫기","interactable":true,"children":[]}]}}
```
- 비활성·`CanvasGroup.alpha=0` 노드는 omniscient 가 아니면 뺀다. `ui_click` 은 hierarchy 노드 id 도 받는다.

### 4.6 `act` → (전투 메시지)* → `action_result`

```json
{"type":"act","seq":14,"action":"interact","args":{"target":"npc_elder","verb":"talk"}}
{"type":"action_result","seq":14,"status":"done","reason":null,"game_seconds":1.8,"observation":{ ...4.5 semantic... }}
```

| `status` | 의미 |
|---|---|
| `done` | 행동 완료 |
| `failed` | 수행 불가. `reason`: `unknown_target`, `not_visible`, `not_interactable`, `out_of_range`, `unreachable`, `blocked_by_modal`, `timeout`(`max_action_seconds` 초과), `invalid_state`. 선택 `reason_detail` |
| `interrupted` | 다른 사건으로 중단. `interrupted_by`: `combat` (0.6.2 는 이것만 쓴다) |

- `action_result` 에 항상 새 `observation` 을 싣는다. `game_seconds` = 이 행동 동안 흐른 세계 시간 (**전투 시간 포함**).
- `reason_detail` 형식: 가림 `"blocked by <ui id>"`, 비활성 `"disabled"`, 범위 밖 `"distance 8.5"`.
- 인자가 틀린 `act`(대상·좌표 둘 다 없음 등) → `error(bad_request)`, 모르는 행동 → `error(unknown_action)`. 이 둘은 `action_result` 가 아니다.
- 행동 규칙:
  - `move_to`: 길찾기(0.6.2 = 격자 A\*). 목표 `interact_range` 안에 들어가면 `done`. 길 없음 → `unreachable`. 시야 밖 대상 → `not_visible`. 이동 중 어그로 → 전투 후 **`interrupted` / `interrupted_by: "combat"`**.
  - `interact`: 범위 밖이면 **자동으로 접근하지 않고** `out_of_range`.
  - `ui_click`: 요소 중심에 **EventSystem 포인터 이벤트**(enter→down→up→click). `onClick.Invoke()` 직접 호출 금지. 가려지면 `failed(not_interactable)` + `"blocked by <id>"`, 비활성이면 `"disabled"`. 사라진 요소 → `unknown_target`.
  - `use_item`: 효과가 없는 상태(예: 체력 가득에서 물약) → `invalid_state`.
  - `wait`: `seconds` 또는 `until` 사건 중 먼저.
  - `engage`: 대상 적에게 다가가 전투 시작. **전투가 끝나면 `done`** (승패는 `combat_ended.result`).
- 모달이 열려 있으면 `move_to`·`interact` → `blocked_by_modal`.

### 4.7 전투: `combat_started` → (`trigger` ↔ `trigger_response`)* → `combat_ended`

`act` 처리 중 게임이 전투를 감지하면 보낸다.

```json
{"type":"combat_started","seq":14,"encounter_id":"enc_1","enemy":"wolf_king","boss":"forest_wolf_king","class":"warrior",
 "scenario":"forest_wolf_king","seed":874512,
 "loadout":{"matched":true,"source":"reports/20261123-001010 (cma)","persona":"stable"},
 "triggers":{"boss_hp_below":[0.75,0.5,0.25]},"start_state":"barrier_inventory_potions",
 "field_state":{"hp_ratio":0.62,"resource_ratio":0.3,"potions":1,"deaths_so_far":0},
 "potions":{"item":"potion_small","available":4,"carried":3,"cap":3}}
{"type":"trigger","seq":14,"encounter_id":"enc_1","index":0,"state":{...},"totals":{...}}
{"type":"trigger_response","encounter_id":"enc_1","index":0,"params":{"w_survival":0.42}}
{"type":"combat_ended","seq":14,"encounter_id":"enc_1",
 "result":{ ...PROTOCOL.md 4.5 episode_result.result 와 같은 필드... },
 "aftermath":{"xp":40,"loot":[{"id":"wolf_fang","count":1}],"respawned_at":null,"potions_used":2}}
```

| 필드 | 규칙 |
|---|---|
| `seq` | 전투를 일으킨 `act` 의 `seq` (`trigger`·`combat_ended` 에도) |
| `encounter_id` | `enc_<k>`, k = 이 게임(reset 이후) 조우 순번, 1부터 |
| `enemy` / `scenario` / `boss` | 적 엔티티 id, 그 적 데이터의 밸런스 시나리오, 시나리오의 보스 |
| `seed` | **조우 seed** = SplitMix64 혼합: `z = reset.seed + 0x9E3779B97F4A7C15·k (mod 2^64)`, `z = (z ^ z>>30)·0xBF58476D1CE4E5B9`, `z = (z ^ z>>27)·0x94D049BB133111EB`, `z ^= z>>31`, **`seed = z & 0x7FFFFFFF`** (구현 `WorldState.EncounterSeed`). 패배 후 다시 싸우면 k 가 늘어 새 seed |
| `loadout` | `matched`(로드아웃 항목 사용 여부), `source`·`persona`(로드아웃 그대로) |
| `triggers` | 이 전투에서 쓰는 트리거 경계, 없으면 `null` |
| `start_state` | `encounter_start_state` 와 같은 값 |
| `potions` | (v0.3) 전투에 가져간 물약. `item` = 퀵슬롯에 등록된 첫 물약 아이템(없으면 `null`), `available` = 그 아이템 인벤토리 수, `cap` = 직업 전투 물약 수(`classes[].stats.potions`, 전사 3), **`carried = min(available, cap)`** = 전투 시작 물약 수 |
| `field_state` | **[0.6.2 미반영]** 보스 결계 회복 **직전** 필드 상태. 전투에는 쓰지 않고 기록만 (v0.3 시작 상태 분포용) |
| `aftermath` | `xp`, `loot:[{id,count}]`, `respawned_at`(패배 시 부활 위치 id, 승리 시 `null`), (v0.3) `potions_used` = 전투에서 마신 수 → 인벤토리에서 그만큼 빠진다 (`item_removed` 사건) |

- **보스 결계 회복 (게임 규칙)**: 조우는 보스 결계 안에서 일어나고, 체력·자원이 학습 판 시작 상태로 맞춰진다.
  - **물약은 인벤토리 것을 쓴다 (v0.3, 사용자 결정 2026-11-23)**: 퀵슬롯에 등록한 물약만 전투에 가져가고(`carried`), 마신 만큼 인벤토리에서 빠진다. 퀵슬롯에 물약이 없으면 물약 0개로 싸운다. 한 번 마실 때 회복량은 **전투 규칙**(직업 `potion_heal_ratio`, 전사 0.4)이고 물약 쿨다운도 전투 규칙 그대로 — 아이템의 필드 회복량(`heal_ratio`)은 필드 `use_item` 에만 쓴다.
  - 트리거 `state.potions_ratio` = 남은 물약 / `cap` (밸런스와 같은 정규화). 학습 판은 항상 `cap` 개로 시작했으므로 `carried < cap` 이면 **학습 분포 밖**이다 (6장).
  - [0.6.2 미반영] 회복 연출·`barrier_restore` 사건·`field_state` 기록. 0.6.2 는 필드에서 체력이 줄 일이 없어(필드 적·함정 피해 없음) 실질 차이는 아직 없다.
- `trigger`·`trigger_response` 의 `state`/`totals`/`params` 규칙은 `PROTOCOL.md` 4.4 와 같다 (`episode_id` 대신 `encounter_id`). 응답 `params` 는 부분 덮어쓰기, `null` = 바꾸지 않음. 전투 시간은 응답까지 정지.
- **잘못된 `trigger_response`** (형식·이름·범위 오류, 다른 index): 0.6.2 는 `error` 를 보내고 **같은 trigger 를 계속 기다린다** (밸런스 4.4 는 판 중단). → **결정 대기** (STATUS 2장). 클라이언트 규칙: 전투 중 받은 `error` 는 클라이언트 버그로 보고 세션을 끝낸다(`play_agent` 구현).
- 전투 중 `trigger_response` 외의 요청 → `error(busy)`, 전투는 계속. 전투 밖 `trigger_response` → `error(bad_request)`.
- **결정성 = 학습과 실전의 동일성**: 같은 `(class, scenario, seed, loadout params, triggers, 트리거 응답)` + **`carried == cap`** → 밸런스 서버 `run_episode` 와 **같은 `result`** (0.6.2 T5 확인). `carried < cap` 이면 물약 수가 달라 결과도 다를 수 있다 (T5 는 물약을 cap 이상 등록한 상태에서 비교).
- 패배 = `cleared:false` 전부 (사망 초과·시간 초과·멈춤). 적의 checkpoint 에서 부활하고 적은 남는다. 승리 = 적 제거·전리품·경험치·처치 목표.
- 전투 후 필드 체력은 가득이다 (`CombatSim` 상태를 필드로 가져오지 않음).

### 4.8 `get_log` → `play_log`

```json
{"type":"get_log","since_seq":0}
{"type":"play_log","entries":[[0.0,"reset",{...}],[1.8,"act",{...}],[1.8,"action_result",{...}]],"next_seq":3}
```
- 세션의 모든 요청·결과·사건. `kind`: `reset`, `act`, `action_result`, `set_loadout`, `combat_start`, `trigger`, `combat_result` + 세계 사건(`dialog_*`, `quest_*`, `item_*`, `gold_changed`, `purchase*`, `zone_entered`, `picked_up`, `respawned` …).
- 게임 쪽 시각 기준 로그가 정본. 같은 `reset.seed` + 같은 행동열 → **같은 로그** (창 ↔ 헤드리스 포함).

### 4.10 `pilot_note` → `pilot_note_ok` (v0.3)

플레이어(LLM — API 루프든 Claude Code 세션이든)가 **무엇을 생각하고 왜 그 행동을 하는지**를 게임 화면에 띄운다. 사람이 창을 보며 QA 플레이를 따라가고, 녹화·GIF 에도 남기기 위한 것.

```json
{"type":"pilot_note","seq":15,"kind":"plan","text":"경비병이 안 보인다. 마을 남동쪽을 더 찾아보자"}
{"type":"pilot_note_ok","seq":15}
```
- `kind`: `thought`(판단) / `plan`(다음에 할 일) / `finding`(버그·개선점 발견). 다른 값 → `error(bad_request)`. `text` 는 한 줄, `max_chars` 이하 (넘으면 클라이언트가 자른다. 게임은 넘친 것을 잘라 표시해도 된다).
- **게임 상태를 바꾸지 않는다**: 세계 시간 0초, 관찰·퀘스트·난수 무관. 행동 사이 언제든 보낼 수 있다 (`reset` 전 → `not_reset`, 전투 중 → `busy`).
- 표시된 메모는 관찰 `screen.ui` 에 **넣지 않는다** (플레이어가 자기 메모를 다시 읽게 되지 않게, 관찰 결정성 유지).
- `play_log` 에 `[t,"pilot_note",{"kind","text"}]` 로 남는다. 재현 비교(같은 seed + 같은 행동열 → 같은 로그)에서는 `pilot_note` 항목을 뺀다.
- 헤드리스(`-batchmode`)에서는 그리지 않고 로그만 남긴다.

### 4.11 `snapshot` / `restore` — 같은 상태에서 다른 행동 시험 (선택, 2026-11-26)

SFT 뒤 학습(RFT 채점, on-policy 선호 쌍)은 **같은 상태에서 후보 행동 여러 개**를 실행해 본다. 선언이 없으면 클라이언트는
후보마다 `reset(seed, class, options)` + 행동열 재생으로 같은 상태를 만든다 (4.4·4.8 결정성) — 맞지만 느리다 (행동 37개 재생 2.2s).
게임이 상태 저장을 지원하면 `play_description.capabilities.snapshot: true` 를 선언한다.

```json
{"type": "snapshot"}                       →  {"type": "snapshot_ok", "id": "snap_3"}
{"type": "restore", "id": "snap_3"}        →  observation (그 시점 관찰, events 는 비움)
```
- 저장 범위 = 결정성에 들어가는 **모든 것**: 세계·퀘스트·인벤토리·골드·창(modal)·난수 상태·조우 순번 k(4.7)·로드아웃. `restore` 뒤 같은 행동 → 저장 직후 그 행동을 한 것과 **같은 결과·같은 로그 꼬리**.
- id 는 연결 안에서만 유효 (연결이 끊기면 사라진다). 개수 상한은 게임이 정하고 넘으면 오래된 것부터 지운다 (지운 id → `error(bad_request)`).
- `reset` 전 → `not_reset`, 전투 중 → `busy`. 헤드리스·창 모두.
- `play_log` 는 restore 하면 그 시점으로 되돌린다 (재현 비교가 깨지지 않게).
- 클라이언트: SDK `btqa_play.branch.Brancher` 가 선언을 보고 snapshot/재생을 고른다. 결과는 같아야 한다 — 다르면 게임 버그.

### 4.9 `error`, `bye`

`PROTOCOL.md` 4.6·4.7 과 같다. 플레이 모드 코드:

| 코드 | 언제 |
|---|---|
| `protocol_version` | `protocol` ≠ `"play"`, 버전 불일치 |
| `bad_request` | JSON 오류, 인자 오류(reset 의 모르는 키·선언하지 않은 `options`, act 인자 없음), 전투 밖 `trigger_response`, 없는 snapshot id |
| `unknown_type` | 모르는 메시지 |
| `not_reset` | `reset` 전에 `act`/`observe` |
| `unknown_action` | `actions` 에 없는 행동 |
| `busy` | 전투 중 `trigger_response` 가 아닌 요청 |
| `unknown_class`, `unknown_scenario`, `unknown_param`, `param_out_of_range` | 밸런스와 같음 |
| `boss_mismatch` | 로드아웃 항목 `boss` ≠ 시나리오 보스 ([0.6.2 미반영] `unknown_boss`) |

`pilot_note` 의 잘못된 `kind`·빈 `text` → `bad_request` (v0.3).

## 5. 클라이언트 — SDK `btqa_play` (+ 쓰는 쪽 클라이언트)

LLM 연결 방식 (사용자 결정 2026-11-22, 2026-11-23 갱신): **본체 = 직접 작성한 tool use 루프 `LlmPilot`, LLM 은 사용자가 각자 고른다** (`LlmBackend` — Claude API / OpenAI 호환 로컬·클라우드). API 없이 대화형 에이전트가 직접 플레이하는 `relay` 도 있다. MCP 래퍼는 선택 (아직 없음).
이유: QA 는 같은 튜토리얼을 자동으로 여러 번 돌려 실패 지점을 통계로 보는 것 — 배치 실행, 전투 대기·막힘 감지·행동/비용 상한, 로그를 코드로 제어해야 한다.

2026-11-24 부터 클라이언트 공통 부분은 **SDK `btqa_play`** (Python ≥ 3.8, 표준 라이브러리만; `anthropic` 백엔드만 anthropic SDK 를 그 안에서 import) 다. 학습 결과 형식에 묶인 부분(로드아웃·전투 응답기)은 SDK 밖, 쓰는 쪽 클라이언트(AI_QA 의 `play_agent`)에 있다.

| SDK `btqa_play` | 역할 |
|---|---|
| `client.py: PlayClient` | 이 프로토콜 클라이언트. `act()` 는 `action_result` 까지 읽으며 전투 메시지를 모으고 `trigger` 는 응답기로 즉답. hello 0.3 → 0.2 → 0.1 |
| `spec.py: PlaySpec` | 외부 플레이 스펙 (`btqa-play-spec/1`) 형식·검사·합치기 |
| `tools.py: build_tools / GameTools` | `play_description.actions` → LLM 도구 (4.2.1: 게임 선언 → 외부 스펙 → 일반형), 도구 호출 실행. `validate_description` = 4.2.1 검사 |
| `prompt.py: build_system_prompt` | SDK 공통 규칙(도구 사용 방식만) + `pilot_guide`(게임 → 스펙) + 클라이언트 추가 규칙 |
| `llm/: LlmBackend` | LLM 확장 지점. `AnthropicBackend`(`output_config.effort`, 프롬프트 캐싱, strict 도구) / `OpenAICompatBackend`(`/v1/chat/completions`, urllib). 설정 `llm.json` + CLI, 키는 환경변수 이름만 |
| `pilot.py: LlmPilot` | 백엔드와 무관한 수동 tool use 루프 (LLM 텍스트 → `pilot_note`). 전투는 도구 실행 안에서 끝나고 `combat_ended` 요약이 도구 결과에 들어간다. 행동 수·진척 없음·호출 상한 |
| `relay.py: PilotRelay` | 대화형 에이전트가 직접 플레이어 (API 호출 없음). 확장 지점 `prepare(client, desc) → 응답기`, `run_meta()` |
| `python -m btqa_play` | `tools`(선언 미리 보기·검사) / `play`(로드아웃 없이 LLM 플레이) / `relay` / `step` |

| 쓰는 쪽 (AI_QA `play_agent/`) | 역할 |
|---|---|
| `loadout.py: LoadoutBuilder` | `reports/*/params.json`(CMA) / `policy.json`(IMPALA) → `set_loadout` 항목. 키 = (class, scenario) |
| `responder.py: CombatResponder / BanditPolicy` | `trigger` 에 즉답 (LLM 거치지 않음). IMPALA 정책 numpy 순전파 |
| `relay.py`, `__main__.py` | SDK `PilotRelay`·`LlmPilot` 에 로드아웃·응답기를 붙인 명령 (`play`·`relay`·`step`·`loadout`) |
| `check.py` | `check-play`: 이 프로토콜 적합성 + QaArena 스크립트 튜토리얼 완주 + 4.2.1 선언 검사 |

- 전투 응답기 = `trigger` 메시지 → `params`(부분 덮어쓰기, `None` = 바꾸지 않음) callable. `on_combat_started(msg)` 가 있으면 조우 시작 때 불린다.
- LLM 은 결정적이지 않다 → 재현은 `play_log` 의 행동열을 그대로 재생해서 한다 (`reset.seed` 동일 + 같은 행동열 → 같은 결과).

## 6. 알려진 제약

- **조우 시작 상태 = 체력·자원 full + 물약은 인벤토리** — 사용자 결정 2026-11-22(결계 full), 2026-11-23(물약은 인벤토리에서, 이 결정이 앞의 것의 물약 부분을 바꾼다).
  - 물약이 `cap` 보다 적게 시작하는 전투는 학습 분포 밖이다. 대응(다음 단계): 밸런스 프로토콜에 `run_episode.initial_state.potions`(SDK 원본 레포, 부 버전 호환)를 추가하고 학습 판 시작 물약을 0~cap 으로 섞어 다시 학습.
  - 그 전 기록: 학습 판은 항상 가득 찬 상태로 시작했으므로 필드 상태를 이어받으면 학습 분포 밖이고 T5 도 깨진다.
  - 다음 계획(이후 버전): `field_state` 로그가 쌓이면 그 분포로 학습 판 시작 상태를 바꿔 다시 학습 → 이어받기 허용. 밸런스 프로토콜에 `run_episode.initial_state` 추가 필요 (1.3, 부 버전 호환). 필드 피해원(일반 적·함정)이 생겨야 의미가 있다.
- **학습하지 않은 보스**는 손 튜닝 default 로 싸운다 (`loadout.matched=false`).
- 계층 정보 기반이라 **시각 UX 결함**(글자 잘림, 겹침, 색 대비)은 잡지 못한다. 0.6.2 시야 판정에 **가림(occlusion) 없음**.
- IMGUI(OnGUI) 로 그린 UI 는 관찰할 수 없다 → 플레이 모드 UI 는 uGUI(TextMeshPro).
- 창 모드는 트리거 응답 대기 중 메인 스레드가 멈춘다 (창이 잠깐 "응답 없음"). 사람 입력 모듈이 없다(결정성).

## 7. 버전

- `protocol:"play"`, `protocol_version` = `주.부`. **0.x 동안은 부 버전 변경도 호환 깨짐** — 주.부가 모두 같아야 접속된다.
- 1.0 확정 시 `PROTOCOL.md` 8장과 같은 규칙.

| 버전 | 날짜 | 내용 |
|---|---|---|
| 0.1 | 2026-11-22 | 초안 |
| 0.3 (추가 2) | 2026-11-26 | **선택 기능만, 버전 그대로**: `reset_options` + `reset.options` (4.4.1, 안 본 시작 조건), `capabilities.snapshot` + `snapshot`/`restore` (4.11, 같은 상태 분기). 선언하지 않은 게임·쓰지 않는 클라이언트 모두 그대로 동작 (클라이언트는 선언했을 때만 보낸다) |
| 0.3 (추가) | 2026-11-24 | **선택 필드만, 버전 그대로**: `actions[].description`·`actions[].schema`, `pilot_guide{system_prompt, about, rules, default_goal}`, `progress_keys` (4.2.1). 보내지 않는 게임·읽지 않는 클라이언트 모두 그대로 동작하므로 hello 버전을 올리지 않는다. 클라이언트 공통 부분 = SDK `btqa_play` (5장) |
| 0.3 | 2026-11-23 | **전투 물약 = 인벤토리**(퀵슬롯 등록분, `min(보유, cap)`, 사용분 차감; `combat_started.potions`, `aftermath.potions_used`, `encounter_start_state:"barrier_inventory_potions"`). `pilot_note`(4.10) + `play_description.pilot_notes`. 클라이언트는 0.3 → 0.2 → 0.1 순으로 hello, 0.3 미만이면 메모를 보내지 않고 `run.json.pilot_notes` 에만 남긴다 |
| 0.2 | 2026-11-23 | 0.6.2 구현 반영: hello 주.부 일치, reset 인자 제한, `scenarios{name,boss}`, `actions` 실제 목록(`ui_input` 제거), `combat.encounters`, 관찰 `quickslots`·퀘스트 `active/ready`·`marker`·`modal:"tip"`, `reason_detail` 형식, `engage`=done / `move_to`=interrupted(combat), 조우 seed 식, `combat_started`·`aftermath` 필드, `set_loadout` 전체 교체, 오류 표 |
