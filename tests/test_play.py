"""BT-QA Play SDK(btqa_play) 자체 테스트 — 가짜 플레이 게임 서버 / 가짜 LLM 으로 (네트워크·LLM 없음)."""
from __future__ import annotations

import ast
import json
import socket
import sys
import tempfile
import threading
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "python"))

from btqa_play import (  # noqa: E402
    SPEC_FORMAT,
    LlmPilot,
    PlaySpec,
    SpecError,
    progress_signature,
    validate_spec,
    PilotConfig,
    PilotRelay,
    PlayClient,
    build_system_prompt,
    build_tools,
    default_goal,
    relay_step,
    tool_sources,
    validate_description,
)
from btqa_play.llm.base import STOP_END, STOP_TOOL_USE, LlmBackend, ToolCall, Turn  # noqa: E402

OBS = {"type": "observation", "seq": 1, "player": {"zone": "village", "gold": 20, "quickslots": [None, None]},
       "quests": [{"id": "q1", "state": "active", "objectives": [{"progress": [0, 1]}]}], "inventory": [], "entities": [],
       "screen": {"modal": None, "ui": []}, "events": []}

GAME_SCHEMA_MOVE = {
    "name": "move_to", "args": {"target": "entity_id | null", "pos": "[x,y,z] | null"},
    "description": "게임이 쓴 이동 설명",
    "schema": {"type": "object", "properties": {
        "target": {"type": "string", "description": "엔티티 id"},
        "pos": {"type": "array", "items": {"type": "number"}, "description": "[x,y,z]"}}, "required": []},
}


class FakePlayGame:
    """1연결 TCP 플레이 서버. actions·pilot_guide 를 바꿔 끼운다."""

    def __init__(self, actions=None, guide=None, versions=("0.3",)):
        self.actions = actions or [{"name": "move_to", "args": {}}, {"name": "interact", "args": {"verb": "talk|pickup"}}]
        self.guide, self.versions = guide, versions
        self.received = []
        self.pending = None
        self.srv = socket.socket()
        self.srv.bind(("127.0.0.1", 0))
        self.srv.listen(1)
        self.port = self.srv.getsockname()[1]
        threading.Thread(target=self._serve, daemon=True).start()

    @property
    def endpoint(self):
        return "127.0.0.1:%d" % self.port

    def _serve(self):
        conn, _ = self.srv.accept()
        f = conn.makefile("rwb")

        def send(m):
            f.write((json.dumps(m, ensure_ascii=False) + "\n").encode("utf-8"))
            f.flush()

        for line in f:
            m = json.loads(line.decode("utf-8"))
            self.received.append(m)
            if m.get("type") == "bye":
                break
            self._handle(m, send)
        conn.close()
        self.srv.close()

    def _handle(self, m, send):
        t = m["type"]
        if t == "hello":
            if m.get("protocol_version") in self.versions:
                send({"type": "hello_ok", "protocol": "play", "protocol_version": m["protocol_version"], "game": "fake", "game_version": "1"})
            else:
                send({"type": "error", "code": "protocol_version"})
        elif t == "play_describe":
            d = {"type": "play_description", "game": "fake", "game_version": "1", "classes": ["c"], "actions": self.actions,
                 "quests": [], "pilot_notes": {"max_chars": 50, "kinds": ["thought", "plan", "finding"]}}
            if self.guide is not None:
                d["pilot_guide"] = self.guide
            send(d)
        elif t == "reset":
            send(dict(OBS))
        elif t == "pilot_note":
            send({"type": "pilot_note_ok", "seq": m["seq"]})
        elif t == "act":
            if m["args"].get("pos") == [0, 0, 32]:
                send({"type": "combat_started", "seq": m["seq"], "encounter_id": "enc_1", "scenario": "s"})
                send({"type": "trigger", "seq": m["seq"], "encounter_id": "enc_1", "index": 0, "state": {}, "totals": {}})
                self.pending = m
                return
            send({"type": "action_result", "seq": m["seq"], "status": "done", "game_seconds": 1.0, "observation": OBS})
        elif t == "trigger_response":
            send({"type": "combat_ended", "seq": self.pending["seq"], "encounter_id": "enc_1", "result": {"cleared": True}, "aftermath": {}})
            send({"type": "action_result", "seq": self.pending["seq"], "status": "interrupted", "interrupted_by": "combat",
                  "game_seconds": 300.0, "observation": OBS})
        elif t == "get_log":
            send({"type": "play_log", "entries": [], "next_seq": 0})


class ScriptedBackend(LlmBackend):
    """정해 둔 Turn 을 차례로 돌려주는 가짜 LLM."""
    label = "fake"

    def __init__(self, turns):
        super().__init__()
        self.turns = list(turns)
        self.system = None
        self.tools = None
        self.results = []

    def start(self, system, tools, first_user):
        self.system, self.tools, self.first = system, tools, json.loads(first_user)

    def complete(self):
        return self.turns.pop(0)

    def add_tool_results(self, results):
        self.results += results

    def add_user(self, text):
        pass


SPEC = {"format": SPEC_FORMAT,
        "actions": {"interact": {"description": "스펙이 쓴 상호작용",
                                 "schema": {"type": "object", "properties": {"target": {"type": "string"},
                                            "verb": {"type": "string", "enum": ["talk", "pickup"]}}}},
                    "move_to": {"description": "스펙 이동 (게임 선언에 밀림)"}},
        "meta_tools": {"finish": {"description": "스펙 finish"}},
        "pilot_guide": {"system_prompt": "관찰 읽는 법", "rules": ["스펙 규칙"], "default_goal": "스펙 목표"},
        "progress_keys": ["player.zone"]}


class ToolsTest(unittest.TestCase):
    def test_no_game_content_in_sdk_generic_fallback(self):
        desc = {"actions": [{"name": "move_to", "args": {"target": "x", "pos": "y"}}, {"name": "dance", "args": {"style": "s"}}]}
        tools = build_tools(desc)
        self.assertEqual([t["name"] for t in tools], ["move_to", "dance", "inspect_ui", "finish"])
        for t in tools:
            s = t["input_schema"]
            self.assertTrue(t["strict"])
            self.assertFalse(s["additionalProperties"])
            self.assertEqual(sorted(s["required"]), sorted(s["properties"]))
        self.assertEqual(tools[0]["input_schema"]["properties"]["target"], {"anyOf": [{"type": "string"}, {"type": "null"}]})
        self.assertEqual(tool_sources(desc), {"move_to": "generic/generic", "dance": "generic/generic"})
        problems = validate_description(desc)
        self.assertTrue(any("move_to: 설명·스키마가" in p for p in problems), problems)   # 일반형은 경고
        self.assertEqual(validate_description(desc, require_text=False), [])

    def test_sdk_module_has_no_game_vocabulary(self):
        src = "".join((ROOT / "python" / "btqa_play" / f).read_text(encoding="utf-8") for f in ("tools.py", "prompt.py", "pilot.py", "actions.py", "record.py", "llm/json_action.py"))
        for word in ("move_to", "interact", "퀘스트", "퀵슬롯", "골드", "물약", "dialog_closed", "talk|"):
            self.assertNotIn(word, src, "SDK 코드에 게임 내용: %s" % word)

    def test_precedence_game_then_spec_then_generic(self):
        spec = PlaySpec.from_dict(SPEC)
        desc = {"actions": [GAME_SCHEMA_MOVE, {"name": "interact", "args": {"target": "t", "verb": "v"}},
                            {"name": "wait", "args": {"seconds": "n"}}]}
        tools = {t["name"]: t for t in build_tools(desc, spec)}
        self.assertEqual(tools["move_to"]["description"], "게임이 쓴 이동 설명")          # 게임이 스펙보다 우선
        self.assertEqual(tools["interact"]["description"], "스펙이 쓴 상호작용")
        self.assertEqual(tools["interact"]["input_schema"]["properties"]["verb"]["enum"], ["talk", "pickup"])
        self.assertEqual(tools["finish"]["description"], "스펙 finish")
        self.assertEqual(tool_sources(desc, spec), {"move_to": "game/game", "interact": "spec/spec", "wait": "generic/generic"})
        self.assertTrue(any(p.startswith("wait:") for p in validate_description(desc, spec)))

    def test_game_declaration_optional_args_become_nullable(self):
        t = build_tools({"actions": [GAME_SCHEMA_MOVE]})[0]
        props = t["input_schema"]["properties"]
        self.assertEqual(props["target"], {"anyOf": [{"type": "string"}, {"type": "null"}], "description": "엔티티 id"})
        self.assertEqual(props["pos"]["anyOf"][1], {"type": "null"})
        self.assertEqual(sorted(t["input_schema"]["required"]), ["pos", "target"])   # strict: 모두 required

    def test_validate_description(self):
        self.assertEqual(validate_description({"actions": [GAME_SCHEMA_MOVE], "pilot_guide": {"about": "x", "rules": ["a"]}}), [])
        bad = validate_description({"actions": [
            {"name": "finish"},
            {"name": "a", "description": ""},
            {"name": "b", "schema": {"type": "object", "properties": {"x": {"type": "str"}}, "required": ["y"]}},
            {"name": "c", "args": {"p": "x"}, "schema": {"type": "object", "properties": {"q": {"type": "string"}}}},
        ], "pilot_guide": {"rules": "하나", "tips": []}}, require_text=False)
        joined = "\n".join(bad)
        for frag in ("finish: 클라이언트 메타 도구", "a: description", "b.x: type", "b: required", "c: args 키",
                     "pilot_guide.rules", "모르는 키 ['tips']"):
            self.assertIn(frag, joined)

    def test_spec_validation(self):
        self.assertEqual(validate_spec(SPEC), [])
        bad = validate_spec({"format": "x", "actions": {"finish": {}, "a": {"schema": {"type": "array"}, "note": 1}},
                             "meta_tools": {"jump": {}}, "pilot_guide": {"foo": 1}, "progress_keys": [], "extra": 1})
        joined = "\n".join(bad)
        for frag in ("format", "모르는 키 extra", "actions.finish", "actions.a: schema", "actions.a: 모르는 키 note",
                     "meta_tools.jump", "pilot_guide: 모르는 키", "progress_keys"):
            self.assertIn(frag, joined)
        with self.assertRaises(SpecError):
            PlaySpec.from_dict({"format": "nope"})
        with tempfile.TemporaryDirectory() as d:
            f = Path(d) / "g.play.json"
            f.write_text(json.dumps(SPEC, ensure_ascii=False), encoding="utf-8")
            self.assertEqual(PlaySpec.load(f).progress_keys, ["player.zone"])

    def test_prompt_core_is_generic_and_guide_merges(self):
        core = build_system_prompt({})
        self.assertNotIn("안내:", core)
        spec = PlaySpec.from_dict(SPEC)
        desc = {"game": "fake", "pilot_guide": {"about": "숲 마을 RPG", "default_goal": "보스를 잡아라"}}
        p = build_system_prompt(desc, spec, extra_rules=["스펙 규칙", "추가 규칙"])
        self.assertIn("관찰 읽는 법", p)                                   # 스펙 system_prompt
        self.assertIn("fake 안내:\n숲 마을 RPG\n- 스펙 규칙\n- 추가 규칙", p)   # 게임 about + 스펙 rules + extra (중복 제거)
        self.assertEqual(default_goal(desc, spec), "보스를 잡아라")          # 게임이 스펙보다 우선
        self.assertEqual(default_goal({}, spec), "스펙 목표")

    def test_progress_signature(self):
        a = dict(OBS, game_time=1.0, events=[[1, "x", {}]], entities=[{"id": "e", "distance": 3.0}])
        b = dict(OBS, seq=9, game_time=5.0, events=[], entities=[{"id": "e", "distance": 1.0}])
        self.assertEqual(progress_signature(a), progress_signature(b))         # 매번 바뀌는 필드는 진척 아님
        c = json.loads(json.dumps(OBS))
        c["player"]["gold"] = 25
        self.assertNotEqual(progress_signature(OBS), progress_signature(c))
        self.assertEqual(progress_signature(OBS, ["player.zone"]), progress_signature(c, ["player.zone"]))


class ClientTest(unittest.TestCase):
    def test_hello_fallback_combat_and_trigger(self):
        g = FakePlayGame(versions=("0.1",))
        with PlayClient.connect(g.endpoint) as c:
            c.hello()
            self.assertEqual(c.protocol_version, "0.1")
            c.describe()
            self.assertFalse(c.note("메모"))           # 0.3 미만 → 보내지 않고 기록만
            r = c.act("move_to", {"pos": [0, 0, 32]}, responder=lambda m: {"w_a": 0.5})
            self.assertEqual(r.status, "interrupted")
            self.assertTrue(r.combats[0].cleared)
        tr = [m for m in g.received if m["type"] == "trigger_response"]
        self.assertEqual(tr[0]["params"], {"w_a": 0.5})
        self.assertFalse([m for m in g.received if m["type"] == "pilot_note"])


class PilotTest(unittest.TestCase):
    def test_pilot_uses_game_guide_and_runs_to_finish(self):
        g = FakePlayGame(actions=[GAME_SCHEMA_MOVE, {"name": "interact", "args": {"target": "t", "verb": "v"}}],
                         guide={"rules": ["게임 규칙 하나"], "default_goal": "늑대를 잡아라"})
        llm = ScriptedBackend([
            Turn(texts=["숲으로"], calls=[ToolCall("1", "move_to", {"target": None, "pos": [0, 0, 32]})], stop=STOP_TOOL_USE),
            Turn(texts=[], calls=[ToolCall("2", "finish", {"success": True, "summary": "ok", "blocked": None})], stop=STOP_TOOL_USE),
        ])
        with PlayClient.connect(g.endpoint) as c:
            c.hello()
            c.describe()
            run = LlmPilot(c, responder=lambda m: None, config=PilotConfig(), backend=llm, spec=SPEC).run(OBS)
        self.assertEqual(run.end_reason, "finish")
        self.assertEqual(run.combats[0]["cleared"], True)
        self.assertEqual(llm.tools[1]["description"], "스펙이 쓴 상호작용")   # 게임이 안 쓴 것은 외부 스펙
        self.assertIn("관찰 읽는 법", llm.system)
        self.assertIn("- 게임 규칙 하나", llm.system)
        self.assertEqual(llm.first["goal"], "늑대를 잡아라")
        self.assertEqual(llm.tools[0]["description"], "게임이 쓴 이동 설명")
        act = next(m for m in g.received if m["type"] == "act")
        self.assertEqual(act["args"], {"pos": [0, 0, 32]})                         # null 인자는 보내지 않는다
        self.assertEqual([m["text"] for m in g.received if m["type"] == "pilot_note"], ["숲으로"])

    def test_text_only_turn_is_nudged(self):
        g = FakePlayGame()
        llm = ScriptedBackend([Turn(texts=["음"], stop=STOP_END),
                               Turn(calls=[ToolCall("2", "finish", {"success": False, "summary": "x", "blocked": None})], stop=STOP_TOOL_USE)])
        with PlayClient.connect(g.endpoint) as c:
            c.hello()
            c.describe()
            run = LlmPilot(c, backend=llm).run(OBS)
        self.assertEqual(run.api_calls, 2)
        self.assertIn({"nudge": True}, run.transcript)


class RelayTest(unittest.TestCase):
    def test_relay_round_trip_with_prepare_hook(self):
        g = FakePlayGame(guide={"default_goal": "목표"})
        with socket.socket() as s:
            s.bind(("127.0.0.1", 0))
            port = s.getsockname()[1]

        class MyRelay(PilotRelay):
            def prepare(self, client, desc):
                self.prepared = desc["game"]
                return lambda m: {"w_a": 1.0}

            def run_meta(self):
                return {"loadout": ["x"]}

        with tempfile.TemporaryDirectory() as d:
            relay = MyRelay(g.endpoint, port=port, out=d)
            th = threading.Thread(target=lambda: setattr(relay, "saved", relay.serve()), daemon=True)
            th.start()
            for _ in range(200):   # relay 가 뜰 때까지
                try:
                    intro = json.loads(relay_step("intro", {}, port=port))
                    break
                except OSError:
                    threading.Event().wait(0.02)
            self.assertEqual(intro["goal"], "목표")
            self.assertIn("system_prompt", intro)
            out = json.loads(relay_step("move_to", {"target": None, "pos": [0, 0, 32]}, port=port, thought="간다", kind="plan"))
            self.assertEqual(out["combats"][0]["cleared"], True)
            relay_step("finish", {"success": True, "summary": "ok", "blocked": None}, port=port)
            th.join(5)
            run = json.loads((relay.saved / "run.json").read_text(encoding="utf-8"))
        self.assertEqual(relay.prepared, "fake")
        self.assertEqual(run["loadout"], ["x"])
        self.assertEqual(run["finish"]["success"], True)
        self.assertEqual([m["params"] for m in g.received if m["type"] == "trigger_response"], [{"w_a": 1.0}])


class StandaloneTest(unittest.TestCase):
    def test_play_sdk_uses_only_stdlib(self):
        allowed = set(sys.builtin_module_names) | {"btqa_play", "__future__", "abc", "argparse", "dataclasses", "json", "os",
                                                    "socket", "sys", "time", "typing", "pathlib", "urllib", "math", "hashlib", "random"}
        lazy = {"anthropic"}   # AnthropicBackend 안에서만
        bad = []
        for path in (ROOT / "python" / "btqa_play").rglob("*.py"):
            tree = ast.parse(path.read_text(encoding="utf-8"))
            for node in ast.walk(tree):
                names = [a.name for a in node.names] if isinstance(node, ast.Import) else \
                    [node.module] if isinstance(node, ast.ImportFrom) and node.module else []
                for n in names:
                    root = n.split(".")[0]
                    if root in lazy and path.name == "anthropic_backend.py":
                        continue
                    if root not in allowed:
                        bad.append("%s: %s" % (path.name, n))
        self.assertEqual(bad, [])

    def test_play_sdk_does_not_import_game_sdk(self):
        for path in (ROOT / "python" / "btqa_play").rglob("*.py"):
            for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
                mods = [a.name for a in node.names] if isinstance(node, ast.Import) else                     [node.module] if isinstance(node, ast.ImportFrom) and node.module else []
                self.assertFalse([m for m in mods if m.split(".")[0] == "btqa_game"], path.name)

    def test_version_matches_repo(self):
        import btqa_game
        import btqa_play
        self.assertEqual(btqa_play.__version__, btqa_game.__version__)


if __name__ == "__main__":
    unittest.main()
