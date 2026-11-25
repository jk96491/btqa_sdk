"""python -m btqa_distill <명령> — 증류 파이프라인 (데이터 만들기는 python -m btqa_play dataset).

    check                                              학습 환경 확인 (torch·CUDA·transformers·peft·llama.cpp·ollama)
    train   --config cfg.json --train train.jsonl [--val val.jsonl]          LoRA 학습 → <out_dir>/adapter
    merge   --base Qwen/Qwen2.5-1.5B-Instruct --adapter <dir> --out <dir>   어댑터 합치기 (HF safetensors)
    gguf    --merged <dir> --out model.gguf --llama-cpp <llama.cpp 폴더> [--quant Q4_K_M]
    ollama  --gguf model.gguf --name qa-qwen1.5b-lora --template-from qwen2.5:1.5b [--gpu]   Ollama 등록 (기본 CPU 전용)
    cpu-variant --base qwen2.5:1.5b --name qwen2.5-1.5b-cpu                  받은 모델의 CPU 전용 변형 (기준선용)
    gpu-variant --base qa-qwen1.5b-r8 --name qa-qwen1.5b-r8-gpu             등록한 모델의 GPU 변형 (같은 가중치, 따로 평가)
    export  --base Qwen/... --adapter <dir> --out-dir <dir> --llama-cpp <dir> --name M [--template-from qwen2.5:1.5b] [--gpu-name M-gpu]
                                                       merge → gguf → ollama 한 번에 (있는 중간 산출물은 건너뜀)
    dpo     --base <SFT merged> --pairs pairs.jsonl --out <dir> [--beta 0.1 --lr 2e-5 --epochs 2]   선호 쌍 DPO → <out>/adapter
    rft     --base <merged> --game H:P --scratch H:P --reward 모듈:함수 [--prepare 모듈:함수] [--spec F] --out <dir>
                                                       GRPO (게임 = 채점기). 보상·판 준비는 게임 쪽 함수 (SDK 는 게임 내용을 모른다)

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


def cmd_gpu_variant(a: argparse.Namespace) -> int:
    from btqa_distill.export import gpu_variant_modelfile, variant_create
    variant_create(a.base, a.name, gpu_variant_modelfile(a.base, a.num_gpu))
    print("등록:", a.name, "= %s (num_gpu %d)" % (a.base, a.num_gpu))
    return 0


def cmd_export(a: argparse.Namespace) -> int:
    from btqa_distill.export import export_all
    print(json.dumps(export_all(a.base, a.adapter, a.out_dir, a.llama_cpp, a.name, template_from=a.template_from, quant=a.quant,
                                gpu_name=a.gpu_name, num_ctx=a.num_ctx), ensure_ascii=False))
    return 0


def cmd_rft(a: argparse.Namespace) -> int:
    from btqa_play.hooks import load_hook
    from btqa_distill.rft import train_rft
    print(train_rft(a.base, a.game, a.scratch, a.seeds, load_hook(a.reward), a.out, prepare=load_hook(a.prepare), spec=a.spec,
                    cls=a.cls, k=a.k, max_steps=a.max_steps, lr=a.lr, kl_coef=a.kl))
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
    p = sub.add_parser("gpu-variant")
    p.add_argument("--base", required=True, help="등록한 Ollama 모델 (CPU 전용)")
    p.add_argument("--name", required=True)
    p.add_argument("--num-gpu", type=int, default=99)
    p.set_defaults(fn=cmd_gpu_variant)
    p = sub.add_parser("export")
    p.add_argument("--base", required=True, help="베이스 HF 모델 (예 Qwen/Qwen2.5-1.5B-Instruct)")
    p.add_argument("--adapter", required=True)
    p.add_argument("--out-dir", required=True, help="merged/·model.<quant>.gguf 를 둘 곳 (git 제외 경로)")
    p.add_argument("--llama-cpp", required=True)
    p.add_argument("--name", required=True, help="Ollama 이름 (CPU 전용)")
    p.add_argument("--template-from")
    p.add_argument("--quant", default="Q4_K_M")
    p.add_argument("--gpu-name", help="GPU 변형도 등록할 이름")
    p.add_argument("--num-ctx", type=int, default=8192)
    p.set_defaults(fn=cmd_export)
    p = sub.add_parser("rft")
    p.add_argument("--base", required=True, help="SFT 를 합친 모델 폴더")
    p.add_argument("--game", required=True, help="본 판 게임")
    p.add_argument("--scratch", required=True, help="분기 채점용 게임 (seed + prefix 재생)")
    p.add_argument("--seeds", type=int, nargs="+", default=[11, 12])
    p.add_argument("--reward", required=True, help="보상 함수 '모듈:함수' (obs_before, tool, input, out) → float")
    p.add_argument("--prepare", help="판 시작 준비 '모듈:함수' (client, describe) → 전투 응답기 등")
    p.add_argument("--spec", help="외부 플레이 스펙 파일")
    p.add_argument("--cls")
    p.add_argument("--k", type=int, default=4)
    p.add_argument("--max-steps", type=int, default=45)
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--kl", type=float, default=0.05)
    p.add_argument("--out", required=True)
    p.set_defaults(fn=cmd_rft)
    a = ap.parse_args(argv)
    return a.fn(a)


if __name__ == "__main__":
    sys.exit(main())
