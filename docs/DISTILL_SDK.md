# 증류 파이프라인 — 큰 LLM(교사)의 플레이 판단 → 작은 로컬 LLM(학생, CPU)

> SDK 0.5.0. 기록·데이터·평가는 `btqa_play`(표준 라이브러리), 학습·내보내기는 **선택 패키지 `btqa_distill`**(torch·transformers·peft).
> 목표는 SFT(+DAgger)에서 멈추는 것이 아니라 **DPO·RFT 까지 이득을 재고 판정할 수 있는 파이프라인** (6장). 이 문서에 게임 내용은 없다 —
> 게임마다 다른 것(평가 조건 값·보상·상황 키·교사 체크 항목)은 게임 쪽 파일·함수로 넘긴다 (7장).
> **모델 가중치·어댑터·GGUF 는 원본이든 증류든 절대 커밋하지 않는다** — `.gitignore`(`*.gguf`·`*.safetensors`·`*.bin`·`adapter_*`·`distill_out/`…), `tests/test_no_model_files.py`, pre-commit hook.

## 1. 흐름

```
교사 플레이 (relay --record)  ─ review/resolve (라벨 검토) ─┐       학생 평가 (eval --suite, steps.jsonl)
        │ btqa-play-step/1 JSONL                            │               │
        └──────────┬────────────────────────────────────────┘               │
  SFT     python -m btqa_play dataset  ──►  train.jsonl / val.jsonl          │
          python -m btqa_distill train ──►  adapter/ (LoRA, GPU) → export (merged·GGUF·Ollama CPU)
  DAgger  relay --student M  (교사가 학생이 간 상태에 정답) → dataset → train   │
  ── 기준 모델 확정: eval --suite → base.json ─────────────────────────────────┘
  DPO     python -m btqa_distill pairs-onpolicy (학생 샘플 K개를 게임으로 채점) → dpo (+chosen NLL) → export → eval → compare
  RFT     python -m btqa_distill rft --suite S --group train (GRPO, k3 KL, SFT 섞기, 판마다 ckpt) → export → eval → compare
```

학습 형식 == 실행 형식: 단계 형식(`btqa_play.actions`)이 system·user 를 만들고, 기록기가 그 글자 그대로 남기고, 학습기가 그대로 쓰고, 실행(`LlmPilot(stateless)` + `JsonActionBackend`)이 같은 함수로 입력을 만든다.

## 2. 데이터 (`btqa_play`)

| 단계 | 명령 |
|---|---|
| 교사 기록 | `python -m btqa_play relay --game H:P --record teacher.jsonl [--reset-options '{..}']` → `step intro` / 매 응답의 `step_input` 만 보고 다음 행동 (`--think` = 목표 thought, `--unsure` = 확신 없는 라벨) |
| 학생 기록 | `python -m btqa_play batch --game H:P --seeds 1-5 --llm openai --model M --tool-mode json --stateless --record` |
| 교정 (DAgger) | `python -m btqa_play relay --game H:P --record dagger.jsonl --student <Ollama 모델>` → `step view` (학생 제안) → 맞으면 `approve [N]`, 틀리면 `step <tool> '<json>' --think ".."` (교사 정답 실행) |
| 데이터셋 | `python -m btqa_play dataset teacher*.jsonl dagger*.jsonl --out data/v1 [--val-runs -s5] [--correction-weight 3] [--prep 모듈:함수]` |
| 자동 교정 (DAgger) | `python -m btqa_play auto --rules 모듈:함수 [--port P]` — 게임 쪽 규칙 `(step_input, 학생 제안) → None(승인) / 정답 dict(교정) / {"defer"}(교사)`. 규칙은 입력만 본다. 경고·defer 에서 멈추면 교사가 `step` 으로 이어 간다 |
| 데이터 조건 검사 | `python -m btqa_play dataset ... --suite S` → `stats.json` `checks` (평가 조건과 같은 판·시작 상태 수, 경고는 stderr) — 5.1장 |
| 라벨 검토 | `python -m btqa_play review teacher*.jsonl dagger*.jsonl` → `resolve <파일> --run R --step N --note ".." [--tool T --input '{..}']` |
| DPO 쌍 (off-policy) | `python -m btqa_play pairs dagger*.jsonl --out pairs.jsonl [--prep 모듈:함수]` (chosen = 교사, rejected = 학생, 행동이 다른 단계만) — 보조용, 기본은 on-policy (6장) |
| 평가 스위트 | `python -m btqa_play suite S` (형식 검사) · `eval --game H:P --suite S --model M [--prepare 모듈:함수]` → `summary.json` + `steps.jsonl` · `compare base.json cand.json` |

**최근 행동 기록 (`--history K`, 선택)**: 기본 단계 입력은 지금 관찰 + 직전 행동 1개뿐이라, 학생은 자기가 같은 행동을 반복하거나 두 행동을 번갈아 하는지 알 수 없다. `--history K` 를 주면 user 맨 앞에 `recent_actions`(그 이전 K개 요약)가 들어간다. **학습과 실행의 K 가 같아야 한다** — `dataset --history K` 는 옛 기록도 판·step 순서로 다시 만든다 (`record.with_history`), `eval`·`relay`·`play`·`rft`·`pairs-onpolicy` 에도 같은 K. 시스템 프롬프트는 바뀌지 않는다.

`dataset` 이 빼는 것: 다른 source(기본 teacher 만), 형식 오류, **grounding 표시**(학생 입력에 없는 id·근거 없는 좌표 = 교사가 기억으로 판단), **해소되지 않은 `needs_review`**(실행해 보니 실패한 정답·진척 없는 반복 — 5장), 같은 입력·같은 목표 중복. `uncertain` 은 남기고 센다 (`--drop-uncertain` 으로 뺀다). `--situation-key 모듈:함수` 를 주면 입력 글자는 달라도 **같은 상황에 다른 도구**를 고른 줄을 `situation_conflicts` 로 센다 (예시 20개까지 `stats.json`). 판(run) 단위로 val 을 뗀다 (`--val-runs` 로 고정 — 교정 판이 검증으로 빠지지 않게). `stats.json` 에 이유별 수.

- **교정 단계** = 학생 제안이 있고 교사 정답과 행동이 다르거나, 기록 뒤 정답을 고친 줄 (`"edited": "<이유>"` — 교사가 틀린 학생 제안을 승인했을 때). `--correction-weight N` 은 그 줄을 train 에 N 번.
- `approve` 는 학생 생각 문장까지 그대로 정답으로 쓴다 → 게임 고유 접두(예: 상황 읽기)가 있다면 `--prep` 훅이 **입력에서 다시 만들게** 한다 (학생이 잘못 옮겨 적은 접두를 배우지 않게).
- **게임 고유 처리는 훅으로** (`btqa_play.hooks.load_hook`, `'모듈:함수'`): SDK 는 그 패키지를 import 하지 않고 실행할 때 이름으로 부른다.

| 훅 | 모양 | 예 (AI_QA) |
|---|---|---|
| `dataset --prep` | SFT 샘플 dict → dict | `play_agent.distill_prep:prep_sample` (목표 thought 앞에 입력에서 옮겨 적은 상황) |
| `pairs --prep` | 쌍 dict → dict | `play_agent.distill_prep:prep_pair` |
| `rft --reward` | `(obs_before, tool, input, out) → float` | `play_agent.distill_reward:reward` |
| `rft --prepare` · `eval --prepare` · `pairs-onpolicy --prepare` | `(PlayClient, describe, **조건의 prepare 인자) → 응답기 또는 None` (판 시작 준비: 로드아웃 등) | `play_agent.distill_reward:prepare` |
| `pairs-onpolicy --reward` | `rft --reward` 와 같다 | `play_agent.distill_reward:reward` |
| `dataset --situation-key` | `user 글 → 상황 키 문자열` | `play_agent.distill_prep:situation_key` |

## 3. 학습·내보내기 (`btqa_distill`, 선택 패키지)

```bash
py -3.12 -m venv venv_distill                                                  # 프로젝트 venv 와 따로 (Python ≥ 3.10)
venv_distill\Scripts\python.exe -m pip install torch --index-url https://download.pytorch.org/whl/cu124
venv_distill\Scripts\python.exe -m pip install -r btqa_sdk/python/btqa_distill/requirements.txt
venv_distill\Scripts\python.exe -m pip install --no-deps -e ./btqa_sdk
venv_distill\Scripts\python.exe -m btqa_distill check [--llama-cpp <llama.cpp>]

venv_distill\Scripts\python.exe -m btqa_distill train --train data/v1/train.jsonl --val data/v1/val.jsonl [--config cfg.json] --out-dir distill_out/r1
venv_distill\Scripts\python.exe -m btqa_distill merge --base Qwen/Qwen2.5-1.5B-Instruct --adapter distill_out/r1/adapter --out distill_out/r1/merged
venv_distill\Scripts\python.exe -m btqa_distill gguf --merged distill_out/r1/merged --out distill_out/r1/model.Q4_K_M.gguf --llama-cpp <llama.cpp>
venv_distill\Scripts\python.exe -m btqa_distill ollama --gguf distill_out/r1/model.Q4_K_M.gguf --name qa-qwen1.5b-r1 --template-from qwen2.5:1.5b
# 위 셋을 한 번에 (있는 중간 산출물은 건너뜀) + GPU 변형
venv_distill\Scripts\python.exe -m btqa_distill export --base Qwen/Qwen2.5-1.5B-Instruct --adapter distill_out/r1/adapter --out-dir distill_out/r1 ^
    --llama-cpp <llama.cpp> --name qa-qwen1.5b-r1 --template-from qwen2.5:1.5b [--gpu-name qa-qwen1.5b-r1-gpu]
# SFT 이후 (6장 절차대로 — 채택은 평가 스위트로)
python -m btqa_play eval --game H:P --suite eval.json --groups train --model M --out reports/train_states   # 학생이 학습 조건에서 간 상태
venv_distill\Scripts\python.exe -m btqa_distill pairs-onpolicy --base distill_out/r1/merged --game H:P2 --records reports/train_states/eval-*/steps.jsonl ^
    --reward mygame.distill:reward [--prepare mygame.distill:prepare] --k 6 --margin 0.5 --out distill_out/data/pairs_onpolicy.jsonl
venv_distill\Scripts\python.exe -m btqa_distill dpo --base distill_out/r1/merged --pairs distill_out/data/pairs_onpolicy.jsonl ^
    --nll-coef 0.2 [--val-pairs V] --out distill_out/r2
venv_distill\Scripts\python.exe -m btqa_distill rft --base distill_out/r1/merged --game H:P --scratch H:P2 --suite eval.json --group train ^
    --reward mygame.distill:reward [--prepare mygame.distill:prepare] --sft distill_out/data/v1/train.jsonl [--spec spec.json] --out distill_out/r3
```

- `TrainConfig` (JSON): `base_model`(기본 `Qwen/Qwen2.5-1.5B-Instruct`), `lora_r` 16, `lora_alpha` 32, `lr` 2e-4, `epochs` 3, `batch_size` 2 × `grad_accum` 8, `max_len` 4096(넘는 샘플은 자르지 않고 뺀다), `dtype` bfloat16, gradient checkpointing.
- loss 는 assistant(JSON 행동)에만. 기반 bf16 + LoRA float32 + autocast (QLoRA 안 씀 — 1.5B~3B 는 12GB 에 들어간다).
- `ollama` 는 기본 **CPU 전용**(`PARAMETER num_gpu 0`) — Ollama 는 GPU 가 있으면 자동으로 GPU 를 쓰므로. 채팅 템플릿은 같은 계열 Ollama 모델에서 복사(`--template-from`).
- 증류 전 기준선용 CPU 변형: `python -m btqa_distill cpu-variant --base qwen2.5:1.5b --name qwen2.5-1.5b-cpu`.
- **평가는 양자화한 모델(Q4)로, 실제로 돌릴 장치에서**: `python -m btqa_play batch ... --model qa-qwen1.5b-r1`. GPU 변형(`gpu-variant` / `export --gpu-name`)은 같은 가중치라도 탐욕 디코딩 갈림길에서 다른 행동을 고를 수 있다 (AI_QA r8: CPU 5/5, GPU 1/5) → 따로 평가.

## 5. 교사 규칙 (게임 무관)

교사(대화형 에이전트·사람)가 relay 로 정답을 달 때. 네가 정한 행동은 그대로 학습 데이터가 된다. 목표는 게임을 잘하는 것이 아니라
**학생이 같은 입력에서 따라 할 수 있는 정답**을 남기는 것이다.

1. **정보** — 판단 근거는 이번 응답의 `step_input` 뿐. 이전 단계에서 본 것·대화 기억·게임 지식은 쓰지 않는다. 입력에 없는 id·좌표·수치는 쓰지 않는다 (`grounding` 이 잡는다). 필요한 정보가 입력에 없으면 **입력만으로 설명되는 탐색 행동**을 고른다.
2. **승인 (DAgger)** — 한 번에 하나만 승인한다 (`approve 1`). 승인 전에 확인한다: ① 인자가 완전한가 (null 대상·빠진 인자) ② 대상 id 가 지금 입력에 있는가 ③ 이 행동이 지금 미완료 목표를 실제로 진전시키는가 (성공만 하고 진척 없는 반복이 아닌가) ④ 게임이 정한 체크 항목 (게임 부록). 하나라도 애매하면 승인하지 말고 교사 정답을 쓴다.
   - 학생 행동이 교사 생각과 달라도 **똑같이 유효하면 승인**한다. 교정은 "실패하거나 진척이 없는 행동"에만.
   - 단, **같은 상황에 이미 쓴 정답이 있으면 그것을 따른다** (일관성이 우선 — 같은 상황에 서로 다른 정답이 쌓이면 작은 학생이 흔들린다. `dataset --situation-key` 가 센다). 다른 유효 행동을 승인하는 것은 그 상황의 첫 정답일 때만.
3. **실행 후 확인** — 응답에 `warning`(needs_review) 이 오면 방금 정답으로 기록한 행동이 실패했거나 진척 없이 반복된 것이다. 다음 단계는 교정하고, 판이 끝나면 `review` → `resolve` 로 그 줄을 해소한다 (맞으면 `--note` 만, 틀렸으면 `--tool --input` 으로 고침 → `edited`, 교정으로 센다). `approve` 는 경고가 오면 멈춘다.
4. **확신이 없으면** `step ... --unsure` — 기록 필드 `uncertain` 으로 남긴다. 생각 문장에 "(불확실)" 같은 글자를 넣지 않는다 (학생이 그 글자까지 배운다).
5. **생각 문장** — 한 문장, 이번 행동의 이유만 (다음 계획 X). 입력에 있는 글·수치를 그대로 옮겨 쓴다. 게임 쪽 전처리(`--prep`)가 붙이는 접두는 쓰지 않는다.
6. **판 끝** — `review` 결과(실패·반복·불확실 수)를 기록과 같이 남긴다.

자동 검사 (SDK): `label_failed`·`repeat_no_progress` 표시(relay), `grounding`(기록기), 같은 입력 다른 행동 `conflicting_inputs`·같은 상황 다른 도구 `situation_conflicts`·생각 문장만 다른 `thought_variants`(dataset).
**"진척"의 뜻은 게임이 정한다** (`progress_keys`, PLAY_PROTOCOL 4.2.1) — SDK 는 그 값이 바뀌었는지만 본다.

### 5.1 DAgger 가 효과를 내는 조건 (게임 무관 체크리스트)

AI_QA 에서 같은 모델·설정에 DAgger 데이터만 바꿔 본 판 완주가 0/5 → 5/5 로 갈렸다 (AI_QA `docs/DISTILL.md` 8.6). 교정 **수**보다 **어떻게 모았는지**가 갈랐다.
새 게임에 적용할 때 아래 넷을 먼저 확인한다.

| 조건 | 확인 방법 | 어겼을 때 본 증상 |
|---|---|---|
| 1. **학생이 실제로 간 상태** (on-policy) | relay `--student` / `auto` — 교사 판만으로 끝내지 않는다 | 교사가 안 가는 장면(학생이 틀려서 들어간 상태)에서 막힌다 |
| 2. **판마다 다른 시작 상태, 평가와 분리** | `dataset --suite S` → `checks` (평가 조건과 같은 판 = 누수 경고, 시작 상태 수). 스위트 `train` 묶음으로만 모은다. 게임 쪽 seed 변화 검사 | 학생이 규칙이 아니라 수치·문장을 외운다. 평가 점수가 실력보다 높게 나온다 (판 N개 = 사실상 1판) |
| 3. **일관되고 검증된 라벨** | 승인은 하나씩 (5장 2), relay 경고(`label_failed`·`repeat_no_progress`) → `review`/`resolve`, `dataset --situation-key` 상황 충돌, (선택) 게임 쪽 규칙 교사 `auto --rules` | 몰아 승인한 틀린 라벨, 비슷한 입력에 다른 정답이 섞여 다른 장면이 새로 흔들린다 |
| 4. **실제로 돌릴 장치·양자화의 학생** | DAgger·평가 모두 같은 장치 (예: CPU Q4) | 같은 가중치가 장치를 바꾸면 갈림길이 뒤집힌다 |

함께 알아 둘 것:
- **입력은 거의 같은데 정답이 다른 장면**은 교정 수를 늘려도 잘 안 고쳐진다. 게임 쪽 전처리(`--prep`)가 그 차이를 **입력에서 옮겨 적는 단서**로 드러내야 한다. 학생이 자기 반복을 모르면 `--history K` (2장).
- **history 를 넣으면 copycat 을 조심한다.** 학생이 기록 속 실패 교대를 그대로 이어 갈 수 있다 (AI_QA r15: seen 완주 1.00 → 0.20). 교정을 바로 실행하면 기록에 실패가 거의 남지 않아 "실패가 쌓인 뒤 빠져나오기" 예시가 없다 → `auto --let-fail N`: 규칙이 교정한 단계에서 정답은 기록하고 게임에는 학생 실수를 N 단계 연속까지 실행한다 (그다음은 정답 실행). history 0 학생에는 필요 없다.
- 교사(대화형 에이전트)는 판 전체를 기억한다. 판 전체는 원인 단계를 찾는 데만 쓰고, 정답은 그 단계 입력만으로 정당화될 때만 붙인다 (5장 1).
- 규칙 교사는 본 장면만 안다. 판을 돌며 규칙이 놓친 장면을 라벨 고침 + 규칙 추가로 메우고, 기존 교사 기록에 규칙을 돌려 오탐이 없는지 확인한다.
- 재학습할 때마다 막히는 장면이 바뀔 수 있다 — 판정은 장면 하나가 아니라 평가 스위트로 (6장).

## 6. SFT 이후: DPO·RFT 를 의미 있게 (게임 무관 절차)

AI_QA 첫 시도(2026-11-25)에서 DPO·RFT 는 모두 후퇴했다. 원인은 알고리즘보다 **측정과 데이터**였다:
기준 모델이 평가 seed 를 이미 다 깼고(오를 곳 없음), seed 가 상황을 거의 바꾸지 않아 5판 = 사실상 1판이었고,
DPO 쌍은 18개·교사 글(학생 분포 밖)이었고, RFT 는 한 판의 신호를 std 로 키워 배웠다. 아래 절차는 그 넷을 막는다.

**0. 전제** — SFT(+DAgger) 모델 = 기준 모델. 후속 학습은 기준 모델이 **아직 못 하는 것**이 있을 때만 의미가 있다. 먼저 기준 모델을 평가 스위트로 잰다.

**1. 포화되지 않은 평가 스위트** (`btqa-eval-suite/1`, `suite.py` — 조건 값은 게임 쪽 파일)
- `seen`: 학습에 쓴 조건 (회귀 확인). 이것만으로는 후속 학습을 판정하지 않는다.
- `robust`: 같은 조건 + `temperature` 샘플 × `repeats` — 탐욕 1판이 아니라 **분포**. 갈림길이 아슬아슬한 모델(장치·양자화에 따라 뒤집히는)은 여기서 드러난다.
- `unseen`: 게임이 정한 안 본 시작 조건 (다른 직업·`reset.options`·seed 범위). 게임이 선언하지 않은 옵션의 조건은 건너뛰고 이유를 남긴다.
- `train`: RFT·on-policy 쌍이 쓰는 학습 조건 — **평가 조건과 seed 를 섞지 않는다**, 평가·비교에서 뺀다.
- 실행 결과에 경고가 나온다: seed 들의 시작 상태가 모두 같음 / 탐욕 궤적이 모두 같음(사실상 1판) / 기준 모델 포화(=1.0). **경고가 있으면 그 묶음으로 이득을 주장하지 않는다** — 게임에 seed 변화·시작 조건을 요청한다 (PLAY_PROTOCOL 4.4.1).
- 기준 모델을 먼저 재고 결과를 고정한다 (`base.json`).

**2. DPO = on-policy 쌍** (`pairs-onpolicy`)
- 학생이 실제로 간 상태에서 — **스위트 `train` 묶음을 학생으로 돌린 기록** (`eval --groups train` 의 `steps.jsonl`). 평가 묶음 기록으로 쌍을 만들면 평가 상태를 학습하게 된다 — 학생 자신의 샘플 K개 → 같은 상태에서 실행해 게임 보상으로 채점 → 최고 − 최저 ≥ margin 인 상태만 (최고, 최저) 쌍. 두 글 모두 학생 글이라 차이는 행동이다. 재생한 관찰이 기록과 다르면 건너뛴다 (`replay_mismatch` — 게임 결정성 확인).
- 학습: `dpo --nll-coef 0.2` (chosen NLL 을 섞어 c·r 확률이 같이 무너지지 않게), 낮은 lr(5e-6), epoch 마다 어댑터 저장. 선호 정확도는 플레이 품질이 아니다.
- 교정 기록 쌍(`pairs`, chosen = 교사)은 보조 (`--include-target` 으로 교사 정답을 후보에 넣어 채점하는 편이 낫다).

**3. RFT** (`rft --suite S --group train`)
- 여러 학습 판(train 조건 전부, `--passes`), 분기는 snapshot(선언 시) 또는 재생.
- 이점 = r − 평균 (`--adv-norm none`, std 로 나누지 않음 — 0 과 0.2 의 차이를 ±1 로 키우지 않게), 탐욕 출력은 그룹 밖(기록·진행용), k3 KL, `--sft train.jsonl` 로 SFT NLL 섞기, 최고−최저 < `--min-gap` 이면 갱신 안 함.
- 보상은 게임 쪽 훅. **진척과 무관한 성공에 보상을 주지 않는다** (보상 해킹: 같은 클릭 반복으로 점수를 모은다). 해킹이 보이면 보상을 고치고 기록한다.
- 판마다 `ckpt-<n>/` — 채택할 체크포인트는 평가 스위트로 고른다 (훈련 보상이 아니다).

**4. 채택** (`compare base.json cand.json`, 스위트의 `adopt`)
- `must_not_regress`(보통 seen) 묶음이 기준 − tolerance 이상 **그리고** `improve_any`(보통 robust·unseen) 중 하나에서 `min_gain` 이상 오를 때만 채택.
- 95% 구간이 겹치면 경고 — 판을 늘려 다시 잰다. 기각해도 기록한다 (실패도 결과다).
- 평가는 실제로 돌릴 장치·양자화로 (CPU Q4 등). 장치를 바꾸면 기준 모델부터 다시 잰다.

## 7. 게임 쪽이 줄 것 (부록)

| 무엇 | 형식 | 비고 |
|---|---|---|
| 평가 조건 | `btqa-eval-suite/1` 파일 (조건 값 = 게임 내용) | 직업·seed 범위·`reset.options`·로드아웃 인자 |
| seed 변화·시작 조건 | 게임 구현 + `reset_options` 선언 (PLAY_PROTOCOL 4.4.1) | 없으면 스위트가 경고·건너뜀 |
| 상태 분기 (선택) | `capabilities.snapshot` (4.11) | 없으면 재생 (느림) |
| 보상 | `reward(obs_before, tool, input, out) → float` 훅 | 진척 사건 중심, 해킹 사례를 주석으로 |
| 판 준비 | `prepare(client, desc, **args)` 훅 | 로드아웃 등 |
| 상황 키 | `situation_key(user_text) → str` 훅 | 교사 일관성 검사 |
| 규칙 교사 (선택) | `rules(step_input, proposal) → None / 정답 / {"defer"}` 훅 | 기계적으로 판단되는 장면의 DAgger 자동 교정 (그 게임에 맞춘 학생일 때) |
| 전처리 | `prep_sample`·`prep_pair` 훅 | 입력에서 옮겨 적는 단서 |
| 교사 체크 항목 | 게임 문서 | 5장 2-④ 에 들어갈 그 게임의 확인 항목 |

## 8. 경계

- `btqa_game`·`btqa_play` = 표준 라이브러리만, `btqa_distill` 을 import 하지 않는다. `btqa_distill` 은 최상위 import 가 표준 라이브러리뿐이고 torch 등은 함수 안에서 (`tests/test_distill.py`).
- SDK 에 게임 내용 없음: 행동 설명·안내는 게임 선언 또는 외부 플레이 스펙 (`PLAY_SDK.md` 2장). 증류의 전처리·보상·판 준비·상황 키·평가 조건도 게임 쪽 함수·파일 → 훅 이름·경로로 (2·7장). 5·6장 규칙은 장르 개념(거리·창·퀘스트 구조 등)을 쓰지 않는다.
- 게임 쪽 패키지는 `btqa_distill`(torch) 을 import 할 필요가 없다 — 보상·전처리는 순수 함수, 학습은 `venv_distill` 에서 SDK 명령으로.
