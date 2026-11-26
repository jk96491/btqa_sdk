"""BT-QA Distill — 교사(큰 LLM)의 플레이 판단을 작은 로컬 LLM 으로 증류 (LoRA) → CPU 실행 모델 (GGUF·Ollama).

btqa_play 가 만든 데이터를 쓴다:
    btqa_play.record   단계 기록 (교사 relay --record / 모델 play --record)
    btqa_play.dataset  기록 → train/val 대화 데이터 (python -m btqa_play dataset)
이 패키지:
    config.TrainConfig   학습 설정 (JSON)
    train.train_lora     LoRA 학습 (GPU 권장) → 어댑터
    export               merge → GGUF(llama.cpp) → 양자화 → Ollama 등록 (CPU 전용, num_gpu 0)
    python -m btqa_distill check|train|merge|gguf|ollama|export|cpu-variant|gpu-variant|dpo|rft

**선택 패키지 — SDK 에서 유일하게 외부 의존성이 있다**: torch(CUDA)·transformers·peft (requirements.txt, Python ≥ 3.10 별도 venv).
최상위 import 는 표준 라이브러리만 (무거운 의존성은 함수 안에서) — btqa_play·btqa_game 은 이 패키지를 import 하지 않는다.
모델 가중치·어댑터·GGUF 는 절대 커밋하지 않는다 (.gitignore·tests/test_no_model_files.py·pre-commit hook).
"""
from btqa_distill.config import QWEN_TARGETS, TrainConfig

__version__ = "0.5.0"   # SDK 레포 버전 (btqa_game·btqa_play 와 같이)

__all__ = ["TrainConfig", "QWEN_TARGETS", "__version__"]
