"""학습 결과 → CPU 실행 모델: LoRA 합치기 → GGUF(llama.cpp) → 양자화 → Ollama 등록 (CPU 전용).

    merge(base, adapter_dir, out_dir)                      # PEFT merge_and_unload → HF safetensors (git 제외 경로)
    to_gguf(merged_dir, out_gguf, llama_cpp, quant="Q4_K_M")
        = python <llama.cpp>/convert_hf_to_gguf.py merged --outtype f16  →  llama-quantize f16.gguf out.gguf Q4_K_M
    write_modelfile(gguf, path, template_from="qwen2.5:1.5b", cpu=True)   # 채팅 템플릿은 같은 계열 Ollama 모델에서 복사
    ollama_create(name, modelfile)

- 평가는 **이 양자화 모델로** 한다 (작은 모델은 양자화로 성능이 변한다).
- cpu=True → PARAMETER num_gpu 0 (사용자 결정: 플레이는 CPU 추론). num_ctx 는 system 고정부 + 한 단계 입력이 들어가게.
- llama.cpp 는 별도로 받는다 (git clone https://github.com/ggml-org/llama.cpp, 변환 스크립트 의존성 + 빌드 또는 릴리스의 llama-quantize).
- 모든 산출물(.safetensors·.gguf)은 git 제외 — 커밋 금지 (tests/test_no_model_files.py, pre-commit hook).
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import List, Optional, Union

PathLike = Union[str, Path]
QUANTS = ("Q4_K_M", "Q5_K_M", "Q8_0", "F16")


def merge(base_model: str, adapter_dir: PathLike, out_dir: PathLike, dtype: str = "float16") -> Path:
    import torch
    from peft import PeftModel
    from transformers import AutoTokenizer

    out = Path(out_dir)
    from btqa_distill.config import load_causal_lm
    model = load_causal_lm(base_model, getattr(torch, dtype))
    model = PeftModel.from_pretrained(model, str(adapter_dir)).merge_and_unload()
    model.save_pretrained(str(out), safe_serialization=True)
    AutoTokenizer.from_pretrained(str(adapter_dir)).save_pretrained(str(out))
    return out


def _find_quantize(llama_cpp: Path) -> Optional[str]:
    for c in ("build/bin/Release/llama-quantize.exe", "build/bin/llama-quantize.exe", "build/bin/llama-quantize",
              "llama-quantize.exe", "llama-quantize"):
        if (llama_cpp / c).exists():
            return str(llama_cpp / c)
    return shutil.which("llama-quantize")


def to_gguf(merged_dir: PathLike, out_gguf: PathLike, llama_cpp: PathLike, quant: str = "Q4_K_M",
            python: Optional[str] = None) -> Path:
    if quant not in QUANTS:
        raise ValueError("quant 는 %s" % "|".join(QUANTS))
    lc, out = Path(llama_cpp), Path(out_gguf)
    conv = lc / "convert_hf_to_gguf.py"
    if not conv.exists():
        raise FileNotFoundError("%s 없음 — llama.cpp 를 clone 하고 경로를 주라" % conv)
    out.parent.mkdir(parents=True, exist_ok=True)
    f16 = out if quant == "F16" else out.with_name(out.stem + ".f16.gguf")
    subprocess.run([python or sys.executable, str(conv), str(merged_dir), "--outfile", str(f16), "--outtype", "f16"], check=True)
    if quant == "F16":
        return out
    q = _find_quantize(lc)
    if not q:
        raise FileNotFoundError("llama-quantize 없음 — llama.cpp 를 빌드하거나 릴리스 바이너리를 PATH 에")
    subprocess.run([q, str(f16), str(out), quant], check=True)
    f16.unlink()
    return out


def ollama_exe() -> str:
    found = shutil.which("ollama")
    if found:
        return found
    win = Path(os.environ.get("LOCALAPPDATA", "")) / "Programs" / "Ollama" / "ollama.exe"
    if win.exists():
        return str(win)
    raise FileNotFoundError("ollama 를 찾을 수 없다 — https://ollama.com/download")


def ollama_template(model: str) -> str:
    """같은 계열 Ollama 모델의 채팅 템플릿 (예 qwen2.5:1.5b) — GGUF 만으로는 Ollama 가 템플릿을 못 고를 수 있다."""
    r = subprocess.run([ollama_exe(), "show", model, "--template"], capture_output=True, text=True, encoding="utf-8")
    if r.returncode != 0 or not r.stdout.strip():
        raise RuntimeError("ollama show %s --template 실패 (먼저 ollama pull %s): %s" % (model, model, r.stderr.strip()))
    return r.stdout


def modelfile_text(gguf: PathLike, template: Optional[str] = None, cpu: bool = True, num_ctx: int = 8192,
                   temperature: float = 0.0) -> str:
    lines: List[str] = ["FROM %s" % Path(gguf).resolve()]
    if template:
        lines.append('TEMPLATE """%s"""' % template.strip("\n"))
    if cpu:
        lines.append("PARAMETER num_gpu 0")   # CPU 추론 (GPU 가 있어도 쓰지 않는다)
    lines += ["PARAMETER num_ctx %d" % num_ctx, "PARAMETER temperature %s" % temperature]
    return "\n".join(lines) + "\n"


def write_modelfile(gguf: PathLike, path: PathLike, template_from: Optional[str] = None, **kw) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(modelfile_text(gguf, ollama_template(template_from) if template_from else None, **kw), encoding="utf-8")
    return p


def cpu_variant_modelfile(base: str, num_ctx: int = 8192) -> str:
    """이미 받은 Ollama 모델(예 qwen2.5:1.5b)의 CPU 전용 변형 — 증류 전 기준선용."""
    return "FROM %s\nPARAMETER num_gpu 0\nPARAMETER num_ctx %d\nPARAMETER temperature 0\n" % (base, num_ctx)


def ollama_create(name: str, modelfile: PathLike) -> None:
    subprocess.run([ollama_exe(), "create", name, "-f", str(modelfile)], check=True)


__all__ = ["merge", "to_gguf", "modelfile_text", "write_modelfile", "cpu_variant_modelfile", "ollama_template",
           "ollama_create", "ollama_exe", "QUANTS"]
