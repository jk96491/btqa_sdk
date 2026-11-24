"""단계 형식(stateless)·JSON 행동 백엔드·단계 기록기·relay 교사 기록 — 가짜 게임 / 가짜 LLM 서버."""
from __future__ import annotations

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
    LlmPilot,
    PilotConfig,
    PilotRelay,
    PlayClient,
    StepRecorder,
    action_schema,
    action_system_prompt,
    build_tools,
    grounding_issues,
    parse_action,
    read_steps,
    relay_step,
    step_input,
)
from btqa_play.llm import make_backend  # noqa: E402
from btqa_play.llm.json_action import JsonActionBackend  # noqa: E402
from btqa_play.tools import compact_observation  # noqa: E402
from tests.test_play import GAME_SCHEMA_MOVE, OBS, SPEC, FakePlayGame  # noqa: E402

TOOLS = build_tools({"actions": [GAME_SCHEMA_MOVE, {"name": "wait", "args": {"seconds": "n"}}]})


class FakePost:
    """OpenAI 호환 서버 흉내: 정해 둔 content 를 차례로 돌려주고 요청 본문을 기록."""

    def __init__(self, contents):
        self.contents = list(contents)
        self.bodies = []

    def __call__(self, url, headers, body, timeout):
        self.bodies.append(json.loads(json.dumps(body)))
        return {"choices": [{"message": {"content": self.contents.pop(0)}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 1000, "completion_tokens": 30}}


class ActionFormatTest(unittest.TestCase):
    def test_system_prompt_lists_tools_and_format(self):
        s = action_system_prompt("규칙", TOOLS)
        self.assertTrue(s.startswith("규칙"))
        self.assertIn("- move_to(target?, pos?): 게임이 쓴 이동 설명", s)
        self.assertIn("- finish(success, summary, blocked?)", s)
        self.assertIn('"thought"', s)

    def test_schema_is_one_variant_per_tool(self):
        sch = action_schema(TOOLS)
        names = [v["properties"]["tool"]["enum"] for v in sch["anyOf"]]
        self.assertEqual(names, [["move_to"], ["wait"], ["inspect_ui"], ["finish"]])
        self.assertNotIn("strict", sch["anyOf"][0]["properties"]["input"])

    def test_parse_action(self):
        names = ["move_to", "finish"]
        act, err = parse_action('```json\n{"thought":"간다","tool":"move_to","input":{"target":"npc"}}\n```', names)
        self.assertIsNone(err)
        self.assertEqual(act, {"thought": "간다", "tool": "move_to", "input": {"target": "npc"}})
        self.assertIn("모르는 도구", parse_action('{"tool":"fly","input":{}}', names)[1])
        self.assertIn("JSON", parse_action("그냥 말", names)[1])

    def test_step_input_has_only_now_and_last_action(self):
        d = json.loads(step_input({"player": {"zone": "a"}}, {"tool": "move_to", "status": "failed", "reason": "not_visible"}))
        self.assertEqual(set(d), {"last_action", "observation"})
        self.assertIsNone(json.loads(step_input({}, None))["last_action"])

    def test_observation_drop(self):
        o = {"type": "observation", "seq": 3, "hidden_included": False,
             "screen": {"ui": [{"id": "t", "role": "text"}, {"id": "b", "role": "button"}]}}
        self.assertEqual(compact_observation(o, ["hidden_included", "screen.ui[role=text]"]),
                         {"screen": {"ui": [{"id": "b", "role": "button"}]}})

    def test_grounding(self):
        user = step_input({"entities": [{"id": "npc_a", "pos": [10, 0, 10]}], "player": {"pos": [0, 0, 0]}})
        obs = json.loads(user)["observation"]
        self.assertEqual(grounding_issues(user, obs, {"target": "npc_a"}), [])
        self.assertEqual(grounding_issues(user, obs, {"target": "npc_b"}), ["id_not_in_input:npc_b"])
        self.assertEqual(grounding_issues(user, obs, {"pos": [11, 0, 9]}), [])
        self.assertEqual(grounding_issues(user, obs, {"pos": [-40, 0, 40]}), ["pos_unsupported:[-40, 0, 40]"])


class JsonBackendTest(unittest.TestCase):
    def test_request_has_schema_and_no_server_tools(self):
        post = FakePost(['{"thought":"간다","tool":"move_to","input":{"target":null,"pos":[0,0,1]}}', "엉망"])
        b = JsonActionBackend("qwen2.5:1.5b", post=post)
        b.start("규칙", TOOLS, step_input({}))
        t = b.complete()
        body = post.bodies[0]
        self.assertNotIn("tools", body)
        self.assertEqual(body["response_format"]["type"], "json_schema")
        self.assertEqual(body["temperature"], 0.0)
        self.assertEqual(body["messages"][0]["content"], action_system_prompt("규칙", TOOLS))
        self.assertEqual((t.calls[0].name, t.calls[0].input, t.texts), ("move_to", {"target": None, "pos": [0, 0, 1]}, ["간다"]))
        b.start("규칙", TOOLS, step_input({}))
        bad = b.complete()
        self.assertIn("출력 형식 오류", bad.calls[0].error)

    def test_make_backend_tool_mode(self):
        self.assertIsInstance(make_backend({"llm": "openai", "model": "m", "tool_mode": "json"}), JsonActionBackend)
        with self.assertRaises(ValueError):
            make_backend({"llm": "openai", "model": "m", "tool_mode": "telepathy"})


class StatelessPilotTest(unittest.TestCase):
    def test_each_step_is_fresh_and_recorded(self):
        g = FakePlayGame(actions=[GAME_SCHEMA_MOVE])
        post = FakePost(['{"thought":"숲","tool":"move_to","input":{"target":null,"pos":[0,0,32]}}',
                         '{"thought":"?","tool":"fly","input":{}}',
                         '{"thought":"끝","tool":"finish","input":{"success":true,"summary":"ok","blocked":null}}'])
        with tempfile.TemporaryDirectory() as d, PlayClient.connect(g.endpoint) as c:
            c.hello()
            c.describe()
            rec = StepRecorder(Path(d) / "s.jsonl", run="t", source="model", model="fake")
            run = LlmPilot(c, responder=lambda m: None, config=PilotConfig(stateless=True), spec=SPEC, recorder=rec,
                           backend=JsonActionBackend("m", post=post, schema=True)).run(OBS)
            steps = read_steps(Path(d) / "s.jsonl")
        self.assertEqual((run.end_reason, run.mode, run.invalid_outputs, run.api_calls), ("finish", "stateless", 1, 3))
        self.assertEqual(len(run.latency_s), 3)
        for b in post.bodies:                                     # 매 요청 = [system, user] 2개 (대화 기록 없음)
            self.assertEqual([m["role"] for m in b["messages"]], ["system", "user"])
        self.assertEqual(len({b["messages"][0]["content"] for b in post.bodies}), 1)   # system 은 판 내내 같은 글자
        u2 = json.loads(post.bodies[1]["messages"][1]["content"])
        self.assertEqual(u2["last_action"]["status"], "interrupted")                  # 직전 행동 결과만
        self.assertEqual(u2["last_action"]["combats"][0]["cleared"], True)
        u3 = json.loads(post.bodies[2]["messages"][1]["content"])
        self.assertIn("모르는 도구", u3["last_action"]["error"])
        self.assertIn("목표: 스펙 목표", post.bodies[0]["messages"][0]["content"])
        self.assertEqual([s["target"]["tool"] for s in steps], ["move_to", "", "finish"])   # 형식 오류 출력은 tool ""
        self.assertIn("fly", steps[1]["result"]["error"])                                 # 원래 출력은 오류에 남는다
        self.assertEqual(steps[0]["messages"], post.bodies[0]["messages"])             # 기록 = 학생 입력 그대로
        self.assertEqual(json.loads(steps[0]["target_text"])["thought"], "숲")
        self.assertEqual(steps[0]["usage"]["input_tokens"], 1000)
        self.assertTrue(steps[0]["grounding"])                                         # 관찰에 위치가 없는 좌표 → 표시


class RelayRecordTest(unittest.TestCase):
    def test_teacher_sees_student_input_and_is_recorded(self):
        g = FakePlayGame(actions=[GAME_SCHEMA_MOVE], guide={"default_goal": "목표"})
        with socket.socket() as s:
            s.bind(("127.0.0.1", 0))
            port = s.getsockname()[1]
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "teacher.jsonl"
            relay = PilotRelay(g.endpoint, port=port, out=d, record=path)
            th = threading.Thread(target=lambda: setattr(relay, "saved", relay.serve()), daemon=True)
            th.start()
            for _ in range(200):
                try:
                    intro = json.loads(relay_step("intro", {}, port=port))
                    break
                except OSError:
                    threading.Event().wait(0.02)
            self.assertIn("출력 형식", intro["system"])
            first = intro["step_input"]
            out = json.loads(relay_step("move_to", {"target": None, "pos": [0, 0, 32]}, port=port, thought="숲으로"))
            self.assertEqual(set(out) - {"grounding"}, {"result", "step_input"})          # 관찰 원문 대신 다음 단계 입력
            self.assertEqual(json.loads(out["step_input"])["last_action"]["status"], "interrupted")
            relay_step("finish", {"success": True, "summary": "ok", "blocked": None}, port=port, thought="끝")
            th.join(5)
            steps = read_steps(path)
            run = json.loads((relay.saved / "run.json").read_text(encoding="utf-8"))
        self.assertEqual(steps[0]["messages"][1]["content"], first)                     # 교사가 본 입력 = 기록된 입력
        self.assertEqual(steps[0]["messages"][0]["content"], intro["system"])
        self.assertEqual((steps[0]["source"], steps[0]["target"]["thought"]), ("teacher", "숲으로"))
        self.assertEqual((run["mode"], run["record_steps"]), ("stateless", 2))


class DaggerRelayTest(unittest.TestCase):
    def test_label_recorded_student_action_executed(self):
        g = FakePlayGame(actions=[GAME_SCHEMA_MOVE, {"name": "wait", "args": {"seconds": "n"}}])
        student_move = '{"thought":"학생은 여기로","tool":"move_to","input":{"target":null,"pos":[0,0,32]}}'
        student = JsonActionBackend("s", post=FakePost([student_move, student_move, student_move]))
        with socket.socket() as s:
            s.bind(("127.0.0.1", 0))
            port = s.getsockname()[1]
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "dagger.jsonl"
            relay = PilotRelay(g.endpoint, port=port, out=d, record=path, student=student)
            th = threading.Thread(target=lambda: setattr(relay, "saved", relay.serve()), daemon=True)
            th.start()
            for _ in range(200):
                try:
                    view = json.loads(relay_step("view", {}, port=port))
                    break
                except OSError:
                    threading.Event().wait(0.02)
            self.assertEqual(view["student"]["tool"], "move_to")                          # 학생 제안이 보인다
            out = json.loads(relay_step("wait", {"seconds": 1, "until": None}, port=port, thought="정답은 기다리기",
                                        execute="student"))
            self.assertEqual(out["result"]["tool"], "move_to")                             # 게임에는 학생 행동
            relay_step("finish", {"success": True, "summary": "ok", "blocked": None}, port=port, thought="끝")
            th.join(5)
            steps = read_steps(path)
        acts = [m for m in g.received if m["type"] == "act"]
        self.assertEqual([a["action"] for a in acts], ["move_to"])
        self.assertEqual((steps[0]["target"]["tool"], steps[0]["target"]["thought"]), ("wait", "정답은 기다리기"))   # 기록은 교사 정답
        self.assertEqual(steps[0]["result"]["executed"], "student")


if __name__ == "__main__":
    unittest.main()
