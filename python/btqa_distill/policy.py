"""학습 중 정책 모델 공통 부품 (DPO·RFT·on-policy 쌍) — LoRA 얹은 모델, 샘플링, 답 구간 log π. torch 는 함수 안에서만.

    pol = LoraPolicy.load("distill_out/r8/merged", lora_r=16, train=True)
    texts = pol.sample(prompt_ids, n=4, temperature=0.9)
    lp = pol.logp(prompt_ids, answer_ids)                       # 어댑터 켠 정책
    with pol.reference(): lr = pol.logp(prompt_ids, answer_ids)  # 어댑터 끈 기준 (= base)
"""
from __future__ import annotations

import contextlib
from typing import Any, Dict, Iterator, List, Optional, Sequence

from btqa_distill.config import QWEN_TARGETS, load_causal_lm
from btqa_distill.train import _ids


class LoraPolicy:
    def __init__(self, model: Any, tok: Any, device: str, dtype: Any, torch: Any) -> None:
        self.model, self.tok, self.device, self.dtype, self.torch = model, tok, device, dtype, torch

    @classmethod
    def load(cls, base: str, lora_r: int = 16, lora_alpha: int = 32, dtype_name: str = "bfloat16",
             train: bool = True) -> "LoraPolicy":
        import torch
        from peft import LoraConfig, get_peft_model
        from transformers import AutoTokenizer
        device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = getattr(torch, dtype_name)
        tok = AutoTokenizer.from_pretrained(base)
        model = load_causal_lm(base, dtype)
        if train:
            model.gradient_checkpointing_enable()
            model.enable_input_require_grads()
            model = get_peft_model(model, LoraConfig(r=lora_r, lora_alpha=lora_alpha, lora_dropout=0.0,
                                                     target_modules=list(QWEN_TARGETS), task_type="CAUSAL_LM"))
            for p in model.parameters():
                if p.requires_grad:
                    p.data = p.data.float()
        model.to(device)
        return cls(model, tok, device, dtype, torch)

    # ------------------------------------------------------------------ 토큰
    def prompt_ids(self, messages: Sequence[Dict[str, str]]) -> List[int]:
        return _ids(self.tok.apply_chat_template([dict(m) for m in messages], tokenize=True, add_generation_prompt=True))

    def answer_ids(self, text: str) -> List[int]:
        return _ids(self.tok(text + (self.tok.eos_token or ""), add_special_tokens=False))

    @property
    def params(self) -> List[Any]:
        return [p for p in self.model.parameters() if p.requires_grad]

    def _autocast(self) -> Any:
        on = self.device.startswith("cuda")
        return self.torch.autocast(device_type="cuda" if on else "cpu", dtype=self.dtype, enabled=on)

    # ------------------------------------------------------------------ 생성·확률
    def sample(self, ids: List[int], n: int = 1, temperature: float = 0.9, top_p: float = 0.95,
               max_new_tokens: int = 160, greedy: bool = False) -> List[str]:
        torch = self.torch
        x = torch.tensor([ids], device=self.device)
        was = self.model.training
        self.model.eval()
        out = []
        with torch.no_grad(), self._autocast():
            for _ in range(1 if greedy else n):
                g = self.model.generate(x, attention_mask=torch.ones_like(x), max_new_tokens=max_new_tokens,
                                        do_sample=not greedy, temperature=None if greedy else temperature,
                                        top_p=None if greedy else top_p,
                                        pad_token_id=self.tok.pad_token_id or self.tok.eos_token_id, use_cache=True)
                out.append(self.tok.decode(g[0, len(ids):], skip_special_tokens=True))
        if was:
            self.model.train()
        return out

    def logp(self, ids: List[int], answer: List[int], mean: bool = False) -> Any:
        """답 구간 log π 합 (mean=True 면 토큰 평균). 답 구간 logits 만 계산 (logits_to_keep, 12GB VRAM)."""
        torch = self.torch
        x = torch.tensor([ids + answer], device=self.device)
        with self._autocast():
            lg = self.model(input_ids=x, logits_to_keep=len(answer) + 1).logits[:, :-1, :].float()
        lp = torch.log_softmax(lg, -1).gather(-1, x[:, len(ids):].unsqueeze(-1)).squeeze(-1)
        return lp.mean() if mean else lp.sum()

    @contextlib.contextmanager
    def reference(self) -> Iterator[None]:
        """어댑터 끈 기준 모델 (π_ref = 시작 모델)."""
        with self.torch.no_grad(), self.model.disable_adapter():
            yield

    def save(self, path: str) -> None:
        self.model.save_pretrained(path)
        self.tok.save_pretrained(path)


def k3_kl(lp: Any, lr: Any, torch: Any) -> Any:
    """KL(π‖π_ref) 의 k3 추정 (π 에서 뽑은 샘플): exp(lr − lp) − (lr − lp) − 1  ≥ 0, 분산이 작다."""
    d = lr - lp
    return torch.exp(d) - d - 1


def sft_nll(pol: LoraPolicy, sample: Dict[str, Any]) -> Any:
    """SFT 샘플 하나의 토큰 평균 NLL (DPO·RFT 에 섞어 SFT 에서 멀어지지 않게)."""
    msgs = sample["messages"]
    return -pol.logp(pol.prompt_ids(msgs[:-1]), pol.answer_ids(msgs[-1]["content"]), mean=True)


__all__ = ["LoraPolicy", "k3_kl", "sft_nll"]
