"""BT-QA Game SDK 자체 테스트 (SDK 폴더만으로 실행: python -m unittest discover -s tests -t .)."""
import ast
import os
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "python"))

from btqa_game import (  # noqa: E402
    BlockDecl,
    ClassDecl,
    DeclarationError,
    GameDecl,
    ParamDecl,
    QaGameServer,
    ScenarioDecl,
    TriggerDecl,
)
from btqa_game.conformance import Conformance, ProbeClient  # noqa: E402
from btqa_game.examples.minimal_arena import MinimalArena  # noqa: E402


def _decl(params, blocks):
    return GameDecl("g", "0", 60.0, 1, (ClassDecl("c", tuple(params), tuple(blocks)),),
                    (ScenarioDecl("base"),), TriggerDecl("boss_hp_below", (0.5,), ("hp_ratio",)))


W = ParamDecl("w_a", 0.0, 1.0, 0.5)
T = ParamDecl("thr", 0.1, 0.9, 0.5, unit="hp_ratio")


class WeightsOnlyRuleTest(unittest.TestCase):
    def test_optimize_comes_from_blocks(self):
        wire = _decl([W, T], [BlockDecl("a", "w_a")]).validate().to_wire()["classes"][0]
        self.assertEqual({p["name"]: p["optimize"] for p in wire["params"]}, {"w_a": True, "thr": False})
        self.assertEqual(wire["blocks"], [{"name": "a", "weight": "w_a"}])
        self.assertEqual([p["unit"] for p in wire["params"]], ["weight", "hp_ratio"])

    def test_rejects_learning_a_non_weight(self):
        with self.assertRaises(DeclarationError):
            _decl([W, ParamDecl("thr", 0.1, 0.9, 0.5, optimize=True)], [BlockDecl("a", "w_a")]).validate()

    def test_rejects_bad_weight_declarations(self):
        cases = (
            ([W], []),                                                               # 블록 없음
            ([W], [BlockDecl("a", "w_missing")]),                                    # 없는 가중치
            ([ParamDecl("w_a", 0.0, 1.0, 0.5, group="persona")], [BlockDecl("a", "w_a")]),
            ([ParamDecl("w_a", -1.0, 1.0, 0.5)], [BlockDecl("a", "w_a")]),            # 음수 범위
            ([W], [BlockDecl("a", "w_a"), BlockDecl("b", "w_a")]),                   # 가중치 공유
            ([ParamDecl("w_a", 0.0, 1.0, 0.5, optimize=False)], [BlockDecl("a", "w_a")]),
        )
        for params, blocks in cases:
            with self.assertRaises(DeclarationError, msg=str((params, blocks))):
                _decl(params, blocks).validate()

    def test_server_refuses_to_start_on_rule_violation(self):
        class Bad(MinimalArena):
            def declaration(self):
                return _decl([W, ParamDecl("thr", 0.1, 0.9, 0.5, optimize=True)], [BlockDecl("a", "w_a")])

        with self.assertRaises(DeclarationError):
            QaGameServer(Bad, port=0)


class ExampleGameTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.server = QaGameServer(MinimalArena, port=0)
        cls.server.start_in_thread()

    @classmethod
    def tearDownClass(cls):
        cls.server.stop()

    def test_conformance_passes(self):
        conf = Conformance(*self.server.address)
        results = conf.run()
        self.assertTrue(conf.passed, "\n".join(r.line() for r in results))
        self.assertTrue(all(r.passed for r in results), "\n".join(r.line() for r in results if not r.passed))

    def test_trigger_response_is_validated(self):
        c = ProbeClient(*self.server.address)
        try:
            with self.assertRaises(Exception) as ctx:
                c.run_episode("duelist", 1, triggers=("enemy_hp_below", [0.5]), on_trigger=lambda m: {"w_attack": 9.0})
            self.assertEqual(getattr(ctx.exception, "code", None), "param_out_of_range")
            r, _ = c.run_episode("duelist", 1)           # 오류 후에도 연결 유지
            self.assertIn("cleared", r)
        finally:
            c.close()

    def test_initial_state_v14(self):
        c = ProbeClient(*self.server.address)
        try:
            d = c.request({"type": "describe"}, "description")
            self.assertEqual([s["name"] for s in d["initial_state"]], ["hp_ratio", "potions_ratio"])
            base, _ = c.run_episode("duelist", 4)
            same, _ = c.run_episode("duelist", 4, initial_state={})          # 빈 객체 = 생략
            self.assertEqual(base, same)
            low, seen = c.run_episode("duelist", 4, initial_state={"hp_ratio": 0.3, "potions_ratio": 0.0},
                                      triggers=("enemy_hp_below", [0.5]))
            self.assertAlmostEqual(seen[0]["state"]["hp_ratio"], 0.3)
            self.assertEqual(seen[0]["state"]["potions_ratio"], 0.0)
            self.assertEqual(low["consumables_used"], 0)                     # 물약 0개로 시작
            for bad, code in (({"mana": 0.5}, "unknown_param"), ({"hp_ratio": 0.01}, "param_out_of_range"),
                              ({"hp_ratio": True}, "bad_request")):
                with self.assertRaises(Exception) as ctx:
                    c.run_episode("duelist", 1, initial_state=bad)
                self.assertEqual(getattr(ctx.exception, "code", None), code, bad)
        finally:
            c.close()

    def test_initial_state_declaration_rules(self):
        from btqa_game import InitialStateDecl
        decl = MinimalArena().declaration()
        bad = GameDecl(**dict(decl.__dict__, initial_state=(InitialStateDecl("mana_ratio"),)))   # state_features 에 없음
        with self.assertRaises(DeclarationError):
            bad.validate()
        bad = GameDecl(**dict(decl.__dict__, initial_state=(InitialStateDecl("hp_ratio", 0.5, 0.4),)))
        with self.assertRaises(DeclarationError):
            bad.validate()


class StandaloneTest(unittest.TestCase):
    def test_python_sdk_uses_only_stdlib(self):
        allowed_roots = set(sys.builtin_module_names) | {"btqa_game", "__future__"}
        stdlib = {"abc", "argparse", "dataclasses", "itertools", "json", "math", "random", "socket",
                  "socketserver", "sys", "threading", "typing"}
        bad = []
        for path in (ROOT / "python" / "btqa_game").rglob("*.py"):
            for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
                names = [a.name for a in node.names] if isinstance(node, ast.Import) else \
                    [node.module] if isinstance(node, ast.ImportFrom) and node.module else []
                for n in names:
                    if n.split(".")[0] not in allowed_roots | stdlib:
                        bad.append("%s: %s" % (path.name, n))
        self.assertEqual(bad, [])

    def test_unity_package_has_no_engine_dependency(self):
        for path in (ROOT / "unity" / "com.btqa.game" / "Runtime").glob("*.cs"):
            self.assertNotIn("using UnityEngine", path.read_text(encoding="utf-8"), path.name)

    def test_versions_agree(self):
        import json

        from btqa_game import PROTOCOL_VERSION, __version__

        pkg = json.loads((ROOT / "unity" / "com.btqa.game" / "package.json").read_text(encoding="utf-8"))
        self.assertEqual(pkg["version"], __version__)
        wire = (ROOT / "unity" / "com.btqa.game" / "Runtime" / "Wire.cs").read_text(encoding="utf-8")
        self.assertIn('ProtocolVersion = "%s"' % PROTOCOL_VERSION, wire)
        self.assertIn('version = "%s"' % __version__, (ROOT / "pyproject.toml").read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
