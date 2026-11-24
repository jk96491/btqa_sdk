"""학습 설정 (표준 라이브러리만 — torch 없이 읽고 검사할 수 있게)."""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Union

#: Qwen2 계열 LoRA 대상 (attention + MLP)
QWEN_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


@dataclass
class TrainConfig:
    base_model: str = "Qwen/Qwen2.5-1.5B-Instruct"   # Hugging Face 이름 또는 로컬 폴더
    out_dir: str = "distill_out/run"                 # 어댑터·로그 (git 제외)
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    target_modules: List[str] = field(default_factory=lambda: list(QWEN_TARGETS))
    lr: float = 2e-4
    epochs: int = 3
    batch_size: int = 2
    grad_accum: int = 8                  # 실효 배치 = batch_size × grad_accum
    warmup_ratio: float = 0.05
    weight_decay: float = 0.0
    max_len: int = 4096                  # 이보다 긴 샘플은 자르지 않고 뺀다 (system 이 잘리면 형식이 깨진다)
    dtype: str = "bfloat16"              # 기반 모델 가중치 (Ampere↑ GPU = bfloat16, 아니면 float16)
    gradient_checkpointing: bool = True
    seed: int = 0
    device: Optional[str] = None         # None = cuda 있으면 cuda

    def validate(self) -> "TrainConfig":
        problems = []
        if self.lora_r <= 0 or self.lora_alpha <= 0:
            problems.append("lora_r·lora_alpha > 0")
        if not 0 <= self.lora_dropout < 1:
            problems.append("0 ≤ lora_dropout < 1")
        if self.lr <= 0 or self.epochs <= 0 or self.batch_size <= 0 or self.grad_accum <= 0:
            problems.append("lr·epochs·batch_size·grad_accum > 0")
        if self.dtype not in ("bfloat16", "float16", "float32"):
            problems.append("dtype 는 bfloat16|float16|float32")
        if self.max_len < 256:
            problems.append("max_len ≥ 256")
        if problems:
            raise ValueError("TrainConfig: " + ", ".join(problems))
        return self

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "TrainConfig":
        unknown = sorted(set(d) - set(cls.__dataclass_fields__) - {k for k in d if k.startswith("_")})
        if unknown:
            raise ValueError("TrainConfig: 모르는 키 %s" % unknown)
        return cls(**{k: v for k, v in d.items() if not k.startswith("_")}).validate()

    @classmethod
    def load(cls, path: Union[str, Path]) -> "TrainConfig":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


__all__ = ["TrainConfig", "QWEN_TARGETS"]
