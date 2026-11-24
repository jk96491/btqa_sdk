"""모델 가중치·어댑터·증류 산출물은 원본이든 증류든 절대 커밋하지 않는다 (사용자 결정 2026-11-24).

- git 이 추적 중(또는 스테이징)인 파일에 모델 확장자가 없어야 한다
- 대표 경로가 .gitignore 로 무시되어야 한다 (규칙이 지워지면 실패)
"""
import shutil
import subprocess
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MODEL_EXT = (".gguf", ".safetensors", ".bin", ".pt", ".pth", ".ckpt", ".onnx", ".ggml", ".lora")
MUST_IGNORE = ("x.gguf", "a/b/model.safetensors", "adapter_model.bin", "distill_out/run1/adapter_config.json",
               "models/qwen/config.json", "checkpoint-100/optimizer.pt", "venv_distill/x")


def _git(*args):
    return subprocess.run(["git", "-C", str(ROOT)] + list(args), capture_output=True, text=True, encoding="utf-8")


@unittest.skipIf(shutil.which("git") is None or not (ROOT / ".git").exists(), "git 저장소가 아님")
class NoModelFilesTest(unittest.TestCase):
    def test_no_tracked_or_staged_model_files(self):
        files = _git("ls-files", "--cached").stdout.splitlines()
        bad = [f for f in files if f.lower().endswith(MODEL_EXT) or f.split("/")[-1].startswith("adapter_model.")]
        self.assertEqual(bad, [], "모델 파일이 git 에 들어 있다 — git rm --cached 로 빼고 .gitignore 확인")

    def test_model_paths_are_ignored(self):
        out = _git("check-ignore", "--no-index", *MUST_IGNORE).stdout.split()
        self.assertEqual(sorted(set(MUST_IGNORE) - set(out)), [], ".gitignore 의 모델 규칙이 빠졌다")


if __name__ == "__main__":
    unittest.main()
