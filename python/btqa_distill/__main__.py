"""python -m btqa_distill <명령> — 증류 파이프라인 (데이터 만들기는 python -m btqa_play dataset).

    check                                              학습 환경 확인 (torch·CUDA·transformers·peft·llama.cpp·ollama)
    train   --config cfg.json --train train.jsonl [--val val.jsonl]          LoRA 학습 → <out_dir>/adapter
    merge   --base Qwen/Qwen2.5-1.5B-Instruct --adapter <dir> --out <dir>   어댑터 합치기 (HF safetensors)
    gguf    --merged <dir> --out model.gguf --llama-cpp <llama.cpp 폴더> [--quant Q4_K_M]
    ollama  --gguf model.gguf --name qa-qwen1.5b-lora --template-from qwen2.5:1.5b [--gpu]   Ollama 등록 (기본 CPU 전용)
    cpu-variant --base qwen2.5:1.5b --name qwen2.5-1.5b-cpu                  받은 모델의 CPU 전용 변형 (기준선용)
    dpo     --base <SFT merged> --pairs pairs.jsonl --out <dir> [--beta 0.1 --lr 2e-5 --epochs 2]   선호 쌍 DPO → <out>/adapter

산출물은 git 제외 경로(distill_out/ 등)에 — 모델 파일은 절대 커밋하지 않는다.
"""
from __future__ import annotations

import argparse
import importlib
import json
import sys
import tempfile
from pathlib import Path
from typing import List, Optional


def _utf8() -> None:
    for s in (sys.stdout, sys.stderr):
        try:
            s.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
        except Exception:
            pass


def cmd_check(a: argparse.Namespace) -> int:
    ok = True
    print("python", sys.version.split()[0], "(학습은 3.10+ 권장)")
    for mod in ("torch", "transformers", "peft"):
        try:
            m = importlib.import_module(mod)
            print("[OK] %s %s" % (mod, getattr(m, "__version__", "?")))
        except ImportError:
            ok = False
            print("[없음] %s — pip install -r btqa_sdk/python/btqa_distill/requirements.txt" % mod)
    try:
        import torch
        print("[%s] CUDA %s" % ("OK" if torch.cuda.is_available() else "없음",
                                torch.cuda.get_device_name(0) if torch.cuda.is_available() else "(CPU 학습은 매우 느리다)"))
    except ImportError:
        pass
    from btqa_distill.export import ollama_exe
    try:
        print("[OK] ollama", ollama_exe())
    except FileNotFoundError as e:
        print("[없음]", e)
    if a.llama_cpp:
        p = Path(a.llama_cpp) / "convert_hf_to_gguf.py"
        print("[%s] llama.cpp %s" % ("OK" if p.exists() else "없음", p))
    return 0 if ok else 1


def cmd_train(a: argparse.Namespace) -> int:
    from btqa_distill.config import TrainConfig
    from btqa_distill.train import train_lora
    cfg = TrainConfig.load(a.config) if a.config else TrainConfig()
    if a.out_dir:
        cfg.out_dir = a.out_dir
    print("어댑터:", train_lora(cfg, a.train, a.val))
    return 0


def cmd_merge(a: argparse.Namespace) -> int:
    from btqa_distill.export import merge
    print("합친 모델:", merge(a.base, a.adapter, a.out))
    return 0


def cmd_gguf(a: argparse.Namespace) -> int:
    from btqa_distill.export import to_gguf
    print("GGUF:", to_gguf(a.merged, a.out, a.llama_cpp, quant=a.quant))
    return 0


def cmd_ollama(a: argparse.Namespace) -> int:
    from btqa_distill.export import ollama_create, write_modelfile
    mf = write_modelfile(a.gguf, a.modelfile or Path(a.gguf).with_suffix(".Modelfile"), template_from=a.template_from,
                         cpu=not a.gpu, num_ctx=a.num_ctx)
    ollama_create(a.name, mf)
    print("등록:", a.name, "(CPU 전용)" if not a.gpu else "(GPU 허용)", "←", mf)
    return 0


def cmd_cpu_variant(a: argparse.Namespace) -> int:
    from btqa_distill.export import cpu_variant_modelfile, ollama_create
    with tempfile.TemporaryDirectory() as d:
        mf = Path(d) / "Modelfile"
        mf.write_text(cpu_variant_modelfile(a.base, a.num_ctx), encoding="utf-8")
        ollama_create(a.name, mf)
    print("등록:", a.name, "= %s (num_gpu 0, num_ctx %d)" % (a.base, a.num_ctx))
    return 0


def cmd_dpo(a: argparse.Namespace) -> int:
    from btqa_distill.dpo import train_dpo
    print(train_dpo(a.base, a.pairs, a.out, beta=a.beta, lr=a.lr, epochs=a.epochs, grad_accum=a.grad_accum))
    return 0


def main(argv: Optional[List[str]] = None) -> int:
    _utf8()
    ap = argparse.ArgumentParser(prog="python -m btqa_distill", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("check")
    p.add_argument("--llama-cpp")
    p.set_defaults(fn=cmd_check)
    p = sub.add_parser("train")
    p.add_argument("--config", help="TrainConfig JSON (없으면 기본: Qwen2.5-1.5B-Instruct, r16)")
    p.add_argument("--train", required=True)
    p.add_argument("--val")
    p.add_argument("--out-dir")
    p.set_defaults(fn=cmd_train)
    p = sub.add_parser("merge")
    p.add_argument("--base", required=True)
    p.add_argument("--adapter", required=True)
    p.add_argument("--out", required=True)
    p.set_defaults(fn=cmd_merge)
    p = sub.add_parser("gguf")
    p.add_argument("--merged", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--llama-cpp", required=True)
    p.add_argument("--quant", default="Q4_K_M")
    p.set_defaults(fn=cmd_gguf)
    p = sub.add_parser("ollama")
    p.add_argument("--gguf", required=True)
    p.add_argument("--name", required=True)
    p.add_argument("--template-from", help="채팅 템플릿을 복사할 같은 계열 Ollama 모델 (예 qwen2.5:1.5b)")
    p.add_argument("--modelfile")
    p.add_argument("--num-ctx", type=int, default=8192)
    p.add_argument("--gpu", action="store_true", help="GPU 허용 (기본은 CPU 전용)")
    p.set_defaults(fn=cmd_ollama)
    p = sub.add_parser("cpu-variant")
    p.add_argument("--base", required=True)
    p.add_argument("--name", required=True)
    p.add_argument("--num-ctx", type=int, default=8192)
    p.set_defaults(fn=cmd_cpu_variant)
    p = sub.add_parser("dpo")
    p.add_argument("--base", required=True, help="SFT 를 합친 모델 폴더 (기준 정책 = 이것)")
    p.add_argument("--pairs", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--beta", type=float, default=0.1)
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--epochs", type=int, default=2)
    p.add_argument("--grad-accum", type=int, default=8)
    p.set_defaults(fn=cmd_dpo)
    a = ap.parse_args(argv)
    return a.fn(a)


if __name__ == "__main__":
    sys.exit(main())
