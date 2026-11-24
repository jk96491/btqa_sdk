"""증류 파이프라인: btqa_play.evaluate·dataset (표준 라이브러리) + btqa_distill 의 torch 없는 부분 + 패키지 경계."""
from __future__ import annotations

import ast
import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "python"))

from btqa_distill.config import TrainConfig  # noqa: E402
from btqa_distill.export import cpu_variant_modelfile, modelfile_text  # noqa: E402
from btqa_distill.train import encode  # noqa: E402
from btqa_play import PilotConfig, PlayClient, StepRecorder, build_dataset, parse_seeds, run_batch  # noqa: E402
from btqa_play.dataset import read_samples  # noqa: E402
from btqa_play.llm.json_action import JsonActionBackend  # noqa: E402
from tests.test_play import GAME_SCHEMA_MOVE, FakePlayGame  # noqa: E402
from tests.test_play_step import FakePost  # noqa: E402


def _rec(rec, run, user, tool="move_to", inp=None, thought="t", result=None, obs=None):
    return rec.write("SYS", user, obs or {"entities": [{"id": "npc", "pos": [0, 0, 0]}]}, thought, tool,
                     inp if inp is not None else {"target": "npc"}, result or {"status": "done"})


class EvaluateTest(unittest.TestCase):
    def test_parse_seeds(self):
        self.assertEqual(parse_seeds("1-3,7, 9"), [1, 2, 3, 7, 9])

    def test_run_batch_summary_and_per_run_usage(self):
        g = FakePlayGame(actions=[GAME_SCHEMA_MOVE])
        finish = '{"thought":"끝","tool":"finish","input":{"success":true,"summary":"ok","blocked":null}}'
        post = FakePost([finish, finish])
        with tempfile.TemporaryDirectory() as d, PlayClient.connect(g.endpoint) as c:
            c.hello()
            c.describe()
            s = run_batch(c, JsonActionBackend("m", post=post), [1, 2], PilotConfig(stateless=True), out_dir=d, record=True,
                          note="x")
            self.assertTrue((Path(d) / "summary.json").exists())
            self.assertEqual(len((Path(d) / "steps.jsonl").read_text(encoding="utf-8").splitlines()), 2)
        self.assertEqual((s["runs"], s["mode"], s["note"], s["invalid_rate"]), (2, "stateless", "x", 0.0))
        self.assertEqual(s["end_reasons"], {"finish": 2})
        self.assertEqual([r["input_tokens_per_call"] for r in s["rows"]], [1000, 1000])   # 판마다 따로 (누적 아님)
        self.assertEqual([m["seed"] for m in g.received if m["type"] == "reset"], [1, 2])


class DatasetTest(unittest.TestCase):
    def test_filters_dedupes_and_splits_by_run(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "steps.jsonl"
            for run in ("r1", "r2", "r3", "r4"):
                rec = StepRecorder(p, run=run, source="teacher")
                _rec(rec, run, '{"o":"%s-1","id":"npc"}' % run)
                _rec(rec, run, '{"o":"%s-1","id":"npc"}' % run)                       # 같은 입력·같은 목표 → 중복
                _rec(rec, run, '{"o":"%s-2"}' % run, inp={"target": "ghost"})         # 입력에 없는 id → ungrounded
            m = StepRecorder(p, run="m1", source="model")
            _rec(m, "m1", '{"o":"m","id":"npc"}')                                     # model 기록 → 기본 제외
            _rec(StepRecorder(p, run="r5", source="teacher"), "r5", '{"o":"x"}', tool="", inp={},
                 result={"error": "출력 형식 오류: x"})
            st = build_dataset([p], Path(d) / "ds", val_ratio=0.25, seed=1)
            train = read_samples(Path(d) / "ds" / "train.jsonl")
            val = read_samples(Path(d) / "ds" / "val.jsonl")
        self.assertEqual(st["dropped"], {"duplicate": 4, "ungrounded": 4, "source:model": 1, "invalid_output": 1})
        self.assertEqual((st["kept"], st["runs"], st["train"], st["val"]), (4, 4, 3, 1))
        self.assertFalse({s["meta"]["run"] for s in train} & {s["meta"]["run"] for s in val})   # 판 단위 분할
        s0 = train[0]
        self.assertEqual([m["role"] for m in s0["messages"]], ["system", "user", "assistant"])
        self.assertEqual(json.loads(s0["messages"][2]["content"])["tool"], "move_to")


class FakeTok:
    """apply_chat_template 흉내: 메시지마다 [role 토큰, 글자 코드..., 끝 토큰]."""
    ROLES = {"system": 1, "user": 2, "assistant": 3}

    def apply_chat_template(self, msgs, tokenize=True, add_generation_prompt=False):
        ids = []
        for m in msgs:
            ids += [self.ROLES[m["role"]]] + [ord(c) % 50 + 10 for c in m["content"]] + [9]
        return ids + ([3] if add_generation_prompt else [])


class DistillPureTest(unittest.TestCase):
    def test_encode_masks_prompt(self):
        sample = {"messages": [{"role": "system", "content": "S"}, {"role": "user", "content": "U"},
                               {"role": "assistant", "content": "AB"}]}
        ids, labels = encode(FakeTok(), sample)
        n_prompt = len(FakeTok().apply_chat_template(sample["messages"][:2], add_generation_prompt=True))
        self.assertEqual(labels[:n_prompt], [-100] * n_prompt)
        self.assertEqual(labels[n_prompt:], ids[n_prompt:])
        self.assertEqual(labels[-1], 9)                                   # 끝 토큰까지 배운다 (생성이 멈춘다)

    def test_config(self):
        self.assertEqual(TrainConfig().base_model, "Qwen/Qwen2.5-1.5B-Instruct")
        with self.assertRaises(ValueError):
            TrainConfig.from_dict({"lr": 0.1, "bogus": 1})
        with self.assertRaises(ValueError):
            TrainConfig(dtype="int3").validate()
        self.assertEqual(TrainConfig.from_dict({"lora_r": 8, "_comment": "x"}).lora_r, 8)

    def test_modelfiles_are_cpu_only_by_default(self):
        t = modelfile_text("m.gguf", template="{{ .Prompt }}")
        self.assertIn("PARAMETER num_gpu 0", t)
        self.assertIn('TEMPLATE """{{ .Prompt }}"""', t)
        self.assertNotIn("num_gpu", modelfile_text("m.gguf", cpu=False))
        self.assertIn("FROM qwen2.5:1.5b\nPARAMETER num_gpu 0", cpu_variant_modelfile("qwen2.5:1.5b"))


class BoundaryTest(unittest.TestCase):
    HEAVY = {"torch", "transformers", "peft", "safetensors", "numpy"}

    def _imports(self, path, top_level_only):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        nodes = tree.body if top_level_only else list(ast.walk(tree))
        for n in nodes:
            if isinstance(n, ast.Import):
                yield from (a.name.split(".")[0] for a in n.names)
            elif isinstance(n, ast.ImportFrom) and n.module:
                yield n.module.split(".")[0]

    def test_play_and_game_never_import_distill(self):
        for pkg in ("btqa_play", "btqa_game"):
            for p in (ROOT / "python" / pkg).rglob("*.py"):
                self.assertNotIn("btqa_distill", set(self._imports(p, False)), p.name)

    def test_distill_heavy_deps_only_inside_functions(self):
        for p in (ROOT / "python" / "btqa_distill").rglob("*.py"):
            self.assertFalse(self.HEAVY & set(self._imports(p, True)), "%s: 최상위에서 무거운 의존성 import" % p.name)
            self.assertNotIn("btqa_game", set(self._imports(p, False)), p.name)

    def test_distill_version(self):
        import btqa_distill
        import btqa_play
        self.assertEqual(btqa_distill.__version__, btqa_play.__version__)


if __name__ == "__main__":
    unittest.main()
