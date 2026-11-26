"""SFT 이후 단계(DPO·RFT)를 위한 SDK 부품 — 라벨 검토·평가 스위트·채택 판정·분기(재생/snapshot)·reset 옵션. 가짜 게임."""
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

from btqa_play import PilotRelay, PlayClient, PlayError, read_steps, relay_step  # noqa: E402
from btqa_play.branch import Brancher, Episode, executed_action, prefixes  # noqa: E402
from btqa_play.dataset import build_dataset  # noqa: E402
from btqa_play.llm.base import LlmBackend, ToolCall, Turn  # noqa: E402
from btqa_play.review import label_flags, resolve, review_list  # noqa: E402
from btqa_play.suite import SUITE_FORMAT, aggregate, compare, conditions, run_suite, validate_suite, wilson  # noqa: E402


class CounterGame:
    """상태 있는 가짜 플레이 서버 (연결마다 스레드). 세계 = 카운터 n, 목표 = n 을 goal 까지.
    행동: step(+1, 진척) / noop(성공, 변화 없음) / fail(failed). reset.options {"start"} 선언, snapshot 선택."""

    def __init__(self, snapshot=False, goal=3):
        self.snapshot, self.goal = snapshot, goal
        self.acts = 0
        self.srv = socket.socket()
        self.srv.bind(("127.0.0.1", 0))
        self.srv.listen(8)
        self.port = self.srv.getsockname()[1]
        threading.Thread(target=self._accept, daemon=True).start()

    @property
    def endpoint(self):
        return "127.0.0.1:%d" % self.port

    def _accept(self):
        while True:
            try:
                conn, _ = self.srv.accept()
            except OSError:
                return
            threading.Thread(target=self._serve, args=(conn,), daemon=True).start()

    def obs(self, st):
        done = st["n"] >= self.goal
        return {"type": "observation", "seq": st["seq"], "player": {"zone": "z%d" % st["seed"]},
                "quests": [] if done else [{"id": "q", "state": "active", "objectives": [{"progress": [st["n"], self.goal]}]}],
                "entities": [], "screen": {"modal": None, "ui": []}, "events": [], "counter": st["n"]}

    def _serve(self, conn):
        f = conn.makefile("rwb")
        st = {"n": 0, "seq": 0, "seed": 1, "log": [], "snaps": {}}

        def send(m):
            f.write((json.dumps(m) + "\n").encode("utf-8"))
            f.flush()

        for line in f:
            m = json.loads(line.decode("utf-8"))
            t = m["type"]
            if t == "bye":
                break
            if t == "hello":
                send({"type": "hello_ok", "protocol": "play", "protocol_version": m["protocol_version"], "game": "counter",
                      "game_version": "1"})
            elif t == "play_describe":
                d = {"type": "play_description", "game": "counter", "game_version": "1", "classes": ["c"],
                     "actions": [{"name": "step", "args": {}}, {"name": "noop", "args": {}}, {"name": "fail", "args": {}}],
                     "quests": [{"id": "q"}], "progress_keys": ["counter"],
                     "reset_options": [{"name": "start", "type": "int", "min": 0, "max": 2}]}
                if self.snapshot:
                    d["capabilities"] = {"snapshot": True}
                send(d)
            elif t == "reset":
                bad = set(m) - {"type", "seed", "class", "options"} or set(m.get("options") or {}) - {"start"}
                if bad:
                    send({"type": "error", "code": "bad_request", "message": str(bad)})
                    continue
                st.update(n=int((m.get("options") or {}).get("start", 0)), seed=int(m.get("seed", 1)), log=[])
                st["seq"] += 1
                send(self.obs(st))
            elif t == "act":
                self.acts += 1
                st["seq"] += 1
                status, ev = "done", []
                if m["action"] == "step":
                    st["n"] += 1
                    if st["n"] == self.goal:
                        st["log"].append([0, "quest_completed", {"quest": "q"}])
                elif m["action"] == "fail":
                    status = "failed"
                send({"type": "action_result", "seq": m["seq"], "status": status, "reason": "nope" if status == "failed" else None,
                      "observation": dict(self.obs(st), events=ev)})
            elif t == "snapshot":
                sid = "s%d" % len(st["snaps"])
                st["snaps"][sid] = dict(n=st["n"], seed=st["seed"], log=list(st["log"]))
                send({"type": "snapshot_ok", "id": sid})
            elif t == "restore":
                st.update(st["snaps"][m["id"]])
                st["log"] = list(st["log"])
                send(self.obs(st))
            elif t == "pilot_note":
                send({"type": "pilot_note_ok", "seq": m.get("seq")})
            elif t == "get_log":
                send({"type": "play_log", "entries": st["log"], "next_seq": len(st["log"])})
        conn.close()

    def close(self):
        self.srv.close()


class PlanBackend(LlmBackend):
    """단계 형식 가짜 LLM: 정해 둔 도구 이름을 차례로 (끝나면 finish)."""
    label = "plan"

    def __init__(self, plan):
        super().__init__()
        self.plan = list(plan)
        self.temperature = 0.0
        self.i = 0

    def start(self, system, tools, first_user):
        pass

    def complete(self):
        name = self.plan[self.i] if self.i < len(self.plan) else "finish"
        self.i += 1
        inp = {"success": True, "summary": "끝", "blocked": None} if name == "finish" else {}
        return Turn(texts=["생각"], calls=[ToolCall(id="c%d" % self.i, name=name, input=inp)])

    def add_tool_results(self, results):
        pass

    def add_user(self, text):
        pass


def _connect(g):
    c = PlayClient.connect(g.endpoint)
    c.hello()
    c.describe()
    return c


class ResetOptionsTest(unittest.TestCase):
    def test_only_declared_options_are_sent(self):
        g = CounterGame()
        with _connect(g) as c:
            self.assertEqual(c.reset_option_names, ["start"])
            self.assertEqual(c.reset(seed=2, options={"start": 2})["counter"], 2)
            with self.assertRaises(PlayError) as e:
                c.reset(seed=2, options={"gold": 5})
            self.assertEqual(e.exception.code, "unknown_reset_option")
        g.close()


class BranchTest(unittest.TestCase):
    def _check(self, snapshot):
        g = CounterGame(snapshot=snapshot)
        br = Brancher(g.endpoint)
        self.assertEqual(br.use_snapshot, snapshot)
        ep = Episode(seed=4, options={"start": 1})
        pre = [("step", {}), ("noop", {})]
        self.assertEqual(br.observation_at(ep, pre)["counter"], 2)
        outs = br.try_actions(ep, pre, [("step", {}), ("fail", {}), ("step", {}), ("noop", {})])
        self.assertEqual([o.get("status") for o in outs], ["done", "failed", "done", "done"])
        self.assertEqual(outs[0]["observation"]["counter"], 3)     # 같은 상태에서 시작
        self.assertEqual(outs[3]["observation"]["counter"], 2)
        replays = br.replays
        br.close()
        g.close()
        return replays

    def test_replay_and_snapshot_give_same_states(self):
        self.assertGreaterEqual(self._check(snapshot=False), 4)   # 후보마다 재생 (같은 행동은 한 번)
        self.assertEqual(self._check(snapshot=True), 1)          # 한 번 재생 → snapshot → restore

    def test_prefixes_from_records(self):
        recs = [{"target": {"tool": "step", "input": {}}, "result": {"tool": "step", "input": {}, "status": "done"}},
                {"target": {"tool": "inspect_ui", "input": {}}, "result": {}},
                {"target": {"tool": "noop", "input": {}}, "result": {"executed": "student", "student_result": {"tool": "fail", "input": {}}}},
                {"target": {"tool": "finish", "input": {}}, "result": {}}]
        self.assertEqual(executed_action(recs[2]), ("fail", {}))   # 교정 모드: 실제 실행은 학생 행동
        self.assertEqual(prefixes(recs), [[], [("step", {})], [("step", {})], [("step", {}), ("fail", {})]])


class LabelReviewTest(unittest.TestCase):
    def test_flags(self):
        self.assertEqual(label_flags(("a", {}), ("a", {}), {"status": "failed"}), ["label_failed"])
        self.assertEqual(label_flags(("a", {}), ("b", {}), {"status": "failed"}), [])             # 교사가 다른 정답 = 교정
        self.assertEqual(label_flags(("a", {}), ("a", {}), {"status": "done"}, ("a", {}), True, progressed=False),
                         ["repeat_no_progress"])
        self.assertEqual(label_flags(("a", {}), ("a", {}), {"status": "done"}, ("a", {}), True, progressed=True), [])

    def test_relay_flags_failed_label_and_dataset_needs_resolution(self):
        g = CounterGame()
        with socket.socket() as s:
            s.bind(("127.0.0.1", 0))
            port = s.getsockname()[1]
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "t.jsonl"
            relay = PilotRelay(g.endpoint, port=port, out=d, record=path, seed=3, options={"start": 1})
            th = threading.Thread(target=lambda: setattr(relay, "saved", relay.serve()), daemon=True)
            th.start()
            for _ in range(200):
                try:
                    relay_step("intro", {}, port=port)
                    break
                except OSError:
                    threading.Event().wait(0.02)
            ok = json.loads(relay_step("step", {}, port=port, thought="하나 올린다"))
            bad = json.loads(relay_step("fail", {}, port=port, thought="이것도"))
            json.loads(relay_step("noop", {}, port=port, thought="가만히", uncertain=True))
            rep = json.loads(relay_step("noop", {}, port=port, thought="또 가만히"))
            relay_step("finish", {"success": True, "summary": "ok", "blocked": None}, port=port, thought="끝")
            th.join(5)
            self.assertNotIn("warning", ok)
            self.assertEqual(bad["warning"]["needs_review"], ["label_failed"])
            self.assertEqual(rep["warning"]["needs_review"], ["repeat_no_progress"])
            steps = read_steps(path)
            self.assertEqual(steps[0]["episode"], {"seed": 3, "options": {"start": 1}})
            self.assertTrue(steps[2]["uncertain"])
            self.assertNotIn("(불확실)", steps[2]["target_text"])                        # 표시는 필드, 글자 아님
            self.assertEqual(len(review_list([path])), 3)
            st = build_dataset([path], Path(d) / "v", val_ratio=0)
            self.assertEqual(st["dropped"].get("needs_review"), 2)
            self.assertEqual(st["uncertain_kept"], 1)
            resolve(path, steps[1]["run"], 2, "실패한 정답 → step 이 맞다", tool="step", tool_input={})
            resolve(path, "-s3", 4, "기다림이 맞는 장면")
            st = build_dataset([path], Path(d) / "v2", val_ratio=0)
            self.assertNotIn("needs_review", st["dropped"])
            fixed = read_steps(path)[1]
            self.assertEqual((fixed["target"]["tool"], fixed["original_target"]["tool"]), ("step", "fail"))
            self.assertEqual(review_list([path], include_uncertain=False), [])
        g.close()

    def test_situation_conflicts(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "s.jsonl"
            rows = []
            for i, (u, tool) in enumerate([('{"z":1,"t":1}', "a"), ('{"z":1,"t":2}', "b"), ('{"z":2,"t":1}', "a")]):
                rows.append({"format": "btqa-play-step/1", "run": "r-s1", "step": i + 1, "source": "teacher",
                             "messages": [{"role": "system", "content": "S"}, {"role": "user", "content": u}],
                             "target": {"thought": "", "tool": tool, "input": {}}, "target_text": "{}", "result": {}, "grounding": []})
            path.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
            st = build_dataset([path], Path(d) / "o", val_ratio=0, situation_key=lambda u: str(json.loads(u)["z"]))
        self.assertEqual(st["conflicting_inputs"], 0)          # 입력 글자는 모두 다르다
        self.assertEqual(st["situation_conflicts"], 1)         # 그러나 상황 z=1 에서 a·b 가 갈린다


SUITE = {"format": SUITE_FORMAT, "name": "t", "defaults": {"max_actions": 10},
         "conditions": [{"name": "seen", "group": "seen", "seeds": "1-3"},
                        {"name": "hard", "group": "unseen", "seeds": "1-2", "options": {"start": 0}},
                        {"name": "later", "group": "unseen", "seeds": "1", "options": {"gold": 1}},
                        {"name": "train", "group": "train", "seeds": "11-12"}],
         "adopt": {"must_not_regress": ["seen"], "improve_any": ["unseen"], "min_gain": 0.1}}


class SuiteTest(unittest.TestCase):
    def test_validate(self):
        self.assertEqual(validate_suite(SUITE), [])
        bad = dict(SUITE, adopt={"improve_any": ["train", "nope"]}, conditions=SUITE["conditions"] + [{"name": "seen", "seeds": "x"}])
        p = validate_suite(bad)
        self.assertTrue(any("train" in x for x in p) and any("nope" in x for x in p) and any("중복" in x for x in p))

    def test_conditions_keep_train_apart(self):
        self.assertEqual([c["name"] for c in conditions(SUITE)], ["seen", "hard", "later"])
        self.assertEqual([c["name"] for c in conditions(SUITE, groups=["train"])], ["train"])

    def test_run_suite_groups_warnings_and_skips(self):
        g = CounterGame()
        with _connect(g) as c, tempfile.TemporaryDirectory() as d:
            class Again(PlanBackend):
                def start(self, system, tools, first_user):
                    if json.loads(first_user)["last_action"] is None:
                        self.i = 0
            s = run_suite(c, Again(["step", "step", "step"]), SUITE, out_dir=d, record=True)
            self.assertTrue((Path(d) / "summary.json").exists())
            steps = read_steps(Path(d) / "steps.jsonl")
        g.close()
        self.assertIn("later", s["skipped"])                                     # 게임이 gold 옵션을 선언하지 않음
        self.assertEqual(set(s["groups"]), {"seen", "unseen"})                   # train 은 평가에서 빠짐
        self.assertEqual((s["groups"]["seen"]["runs"], s["groups"]["seen"]["success"]), (3, 1.0))
        self.assertTrue(any("궤적이 모두 같다" in w for w in s["warnings"]))     # seed 가 판을 바꾸지 않으면 경고
        self.assertEqual(steps[0]["episode"]["seed"], 1)

    def test_stats(self):
        self.assertEqual(wilson(5, 5)[1], 1.0)
        self.assertLess(wilson(5, 5)[0], 0.6)                                   # 5/5 의 하한은 0.57 — 판 수가 적다
        a = aggregate([{"success": True, "progress": 1.0, "trajectory": "x"}, {"success": False, "progress": 0.5, "trajectory": "y"}])
        self.assertEqual((a["success"], a["progress"], a["distinct_trajectories"]), (0.5, 0.75, 2))

    def test_compare(self):
        def res(seen, unseen):
            return {"groups": {"seen": {"progress": seen, "progress_ci": [seen - .1, seen]},
                               "unseen": {"progress": unseen, "progress_ci": [unseen - .1, unseen + .1]}}, "adopt": SUITE["adopt"]}
        self.assertTrue(compare(res(1.0, 0.4), res(1.0, 0.7))["adopt"])
        r = compare(res(1.0, 0.4), res(0.8, 0.9))
        self.assertFalse(r["adopt"])
        self.assertTrue(any("후퇴" in x for x in r["reasons"]))
        self.assertFalse(compare(res(1.0, 0.4), res(1.0, 0.45))["adopt"])        # 이득 부족
        sat = compare(res(1.0, 1.0), res(1.0, 1.0), {"improve_any": ["seen"]})
        self.assertTrue(any("포화" in w for w in sat["warnings"]))


class RelayAutoTest(unittest.TestCase):
    def test_rules_correct_approve_and_defer(self):
        from btqa_play.llm.json_action import JsonActionBackend
        from btqa_play.relay import relay_auto
        from tests.test_play_step import FakePost
        g = CounterGame(goal=5)
        outs = ['{"thought":"올린다","tool":"step","input":{}}', '{"thought":"망한다","tool":"fail","input":{}}',
                '{"thought":"올린다","tool":"step","input":{}}', '{"thought":"쉰다","tool":"noop","input":{}}']
        student = JsonActionBackend("s", post=FakePost(outs * 3))
        with socket.socket() as s:
            s.bind(("127.0.0.1", 0))
            port = s.getsockname()[1]

        def rules(user, st):
            if st["tool"] == "fail":
                return {"tool": "step", "input": {}, "thought": "실패 대신 올린다", "rule": "no_fail"}
            if st["tool"] == "noop":
                return {"defer": "쉬기는 교사가 본다"}
            return None
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "auto.jsonl"
            relay = PilotRelay(g.endpoint, port=port, out=d, record=path, student=student)
            th = threading.Thread(target=relay.serve, daemon=True)
            th.start()
            for _ in range(200):
                try:
                    relay_step("intro", {}, port=port)
                    break
                except OSError:
                    threading.Event().wait(0.02)
            r = relay_auto(rules, max_steps=10, port=port)
            relay_step("finish", {"success": True, "summary": "ok", "blocked": None}, port=port, thought="끝")
            th.join(5)
            steps = read_steps(path)
        g.close()
        self.assertEqual((r["approved"], r["corrected"], r["rules"]), (2, 1, {"no_fail": 1}))
        self.assertTrue(r["stop"].startswith("교사 필요"))
        self.assertEqual([x["target"]["tool"] for x in steps[:3]], ["step", "step", "step"])     # 교정 줄은 규칙의 정답
        self.assertEqual(steps[1]["student"]["tool"], "fail")                                    # 학생 제안은 남는다 (교정으로 셈)


if __name__ == "__main__":
    unittest.main()


from btqa_play.actions import ActionHistory, history_entry, step_input  # noqa: E402
from btqa_play.record import with_history  # noqa: E402


class HistoryTest(unittest.TestCase):
    """단계 입력의 recent_actions (history=K): 실행 쪽(ActionHistory)과 데이터 쪽(with_history)이 같은 글자를 만든다."""

    def test_action_history_window(self):
        acts = [{"tool": "t%d" % i, "input": {"x": i}, "status": "done", "reason_detail": "길다"} for i in range(4)]
        h = ActionHistory(2)
        self.assertEqual(json.loads(h.user({}))["recent_actions"], [])          # 첫 단계도 같은 모양
        for x in acts:
            h.push(x)
        d = json.loads(h.user({"a": 1}))
        self.assertEqual(d["last_action"], acts[3])                              # 직전 행동은 자세히
        self.assertEqual(d["recent_actions"], [history_entry(acts[1]), history_entry(acts[2])])   # 그 이전 2개는 요약, 오래된 것부터
        self.assertNotIn("reason_detail", d["recent_actions"][0])
        self.assertEqual(list(d), ["recent_actions", "last_action", "observation"])
        h0 = ActionHistory(0)
        h0.push(acts[0])
        self.assertEqual(h0.user({}), step_input({}, acts[0]))                  # K=0 = 옛 형식 그대로

    def test_rebuilt_records_match_live_input(self):
        g = CounterGame()
        with _connect(g) as c, tempfile.TemporaryDirectory() as d:
            class Again(PlanBackend):
                def start(self, system, tools, first_user):
                    if json.loads(first_user)["last_action"] is None:
                        self.i = 0
            run_suite(c, Again(["step"] * 5), dict(SUITE, conditions=SUITE["conditions"][:1]), out_dir=d, record=True, history=2)
            live = read_steps(Path(d) / "steps.jsonl")
        g.close()
        self.assertIn("recent_actions", json.loads(live[-1]["messages"][1]["content"]))
        self.assertEqual(len(json.loads(live[3]["messages"][1]["content"])["recent_actions"]), 2)
        old = with_history(live, 0)                                              # 옛 형식 기록 (recent_actions 없음)
        self.assertNotIn("recent_actions", json.loads(old[3]["messages"][1]["content"]))
        rebuilt = with_history(old, 2)
        self.assertEqual([r["messages"][1]["content"] for r in rebuilt], [r["messages"][1]["content"] for r in live])

    def test_gap_in_steps_resets_history(self):
        def rec(step, last):
            return {"run": "r", "step": step, "messages": [{"role": "system", "content": "s"},
                                                           {"role": "user", "content": step_input({"n": step}, last)}]}
        a = {"tool": "a", "input": {}, "status": "done"}
        out = with_history([rec(1, None), rec(2, a), rec(3, a), rec(7, a), rec(8, a)], 3)
        n = [len(json.loads(r["messages"][1]["content"])["recent_actions"]) for r in out]
        self.assertEqual(n, [0, 0, 1, 0, 1])                                     # 4~6 이 빠졌으면 그 이전은 모른다
