# 증류 파이프라인 — 큰 LLM(교사)의 플레이 판단 → 작은 로컬 LLM(학생, CPU)

> SDK 0.4.0. 기록·데이터·평가는 `btqa_play`(표준 라이브러리), 학습·내보내기는 **선택 패키지 `btqa_distill`**(torch·transformers·peft).
> **모델 가중치·어댑터·GGUF 는 원본이든 증류든 절대 커밋하지 않는다** — `.gitignore`(`*.gguf`·`*.safetensors`·`*.bin`·`adapter_*`·`distill_out/`…), `tests/test_no_model_files.py`, pre-commit hook.

## 1. 흐름

```
교사 플레이 (relay --record)        학생 평가 (batch --record)
        │ btqa-play-step/1 JSONL            │ (같은 형식, source=model)
        └──────────┬────────────────────────┘
          python -m btqa_play dataset  ──►  train.jsonl / val.jsonl   (btqa-play-sft/1: system·user·assistant)
          python -m btqa_distill train ──►  adapter/                  (LoRA, GPU)
          python -m btqa_distill merge ──►  merged/ (safetensors)
          python -m btqa_distill gguf  ──►  model.Q4_K_M.gguf         (llama.cpp)
          python -m btqa_distill ollama ─►  Ollama 모델 (CPU 전용, num_gpu 0)
          python -m btqa_play batch    ──►  summary.json (증류 전후 비교)
```

학습 형식 == 실행 형식: 단계 형식(`btqa_play.actions`)이 system·user 를 만들고, 기록기가 그 글자 그대로 남기고, 학습기가 그대로 쓰고, 실행(`LlmPilot(stateless)` + `JsonActionBackend`)이 같은 함수로 입력을 만든다.

## 2. 데이터 (`btqa_play`)

| 단계 | 명령 |
|---|---|
| 교사 기록 | `python -m btqa_play relay --game H:P --record teacher.jsonl` → `step intro` / 매 응답의 `step_input` 만 보고 다음 행동 (`--think` = 목표 thought) |
| 학생 기록 | `python -m btqa_play batch --game H:P --seeds 1-5 --llm openai --model M --tool-mode json --stateless --record` |
| 데이터셋 | `python -m btqa_play dataset teacher*.jsonl --out data/v1 [--val-ratio 0.1] [--sources teacher]` |

`dataset` 이 빼는 것: 다른 source(기본 teacher 만), 형식 오류, **grounding 표시**(학생 입력에 없는 id·근거 없는 좌표 = 교사가 기억으로 판단), 같은 입력·같은 목표 중복. 판(run) 단위로 val 을 뗀다. `stats.json` 에 이유별 수.
교정(DAgger) 데이터: 학생이 막힌 상태에서 교사가 relay 로 이어서 플레이하면 같은 형식(source=teacher)으로 쌓인다.

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
```

- `TrainConfig` (JSON): `base_model`(기본 `Qwen/Qwen2.5-1.5B-Instruct`), `lora_r` 16, `lora_alpha` 32, `lr` 2e-4, `epochs` 3, `batch_size` 2 × `grad_accum` 8, `max_len` 4096(넘는 샘플은 자르지 않고 뺀다), `dtype` bfloat16, gradient checkpointing.
- loss 는 assistant(JSON 행동)에만. 기반 bf16 + LoRA float32 + autocast (QLoRA 안 씀 — 1.5B~3B 는 12GB 에 들어간다).
- `ollama` 는 기본 **CPU 전용**(`PARAMETER num_gpu 0`) — Ollama 는 GPU 가 있으면 자동으로 GPU 를 쓰므로. 채팅 템플릿은 같은 계열 Ollama 모델에서 복사(`--template-from`).
- 증류 전 기준선용 CPU 변형: `python -m btqa_distill cpu-variant --base qwen2.5:1.5b --name qwen2.5-1.5b-cpu`.
- **평가는 양자화한 모델(Q4)로**: `python -m btqa_play batch ... --model qa-qwen1.5b-r1`.

## 4. 경계

- `btqa_game`·`btqa_play` = 표준 라이브러리만, `btqa_distill` 을 import 하지 않는다. `btqa_distill` 은 최상위 import 가 표준 라이브러리뿐이고 torch 등은 함수 안에서 (`tests/test_distill.py`).
- SDK 에 게임 내용 없음: 행동 설명·안내는 게임 선언 또는 외부 플레이 스펙 (`PLAY_SDK.md` 2장).
