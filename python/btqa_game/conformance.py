"""프로토콜 적합성 검사 — 게임 쪽 SDK 에 들어 있는 도구 (표준 라이브러리만, torch·학습 플랫폼 불필요).

게임 개발자가 자기 게임 서버를 띄우고 돌린다:

    python -m btqa_game check --game 127.0.0.1:7777

학습 플랫폼(balance_qa)의 `check-game` 과 같은 항목을 검사한다. 이 검사를 통과하면 학습 플랫폼에 붙는다.
"""
from __future__ import annotations

import itertools
import json
import math
import socket
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

from btqa_game import protocol as P

CHECK_SEED = 424242
REQUIRED_TOTALS = ("elapsed", "deaths", "consumables_used")
_ids = itertools.count(1)


class GameErrorReply(Exception):
    def __init__(self, code: str, message: str) -> None:
        super().__init__("%s: %s" % (code, message))
        self.code = code


class ProbeClient:
    """검사용 최소 클라이언트 (연결 1개, 요청 순차)."""

    def __init__(self, host: str, port: int, timeout: float = 60.0) -> None:
        self.sock = socket.create_connection((host, port), timeout=timeout)
        self.sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        self.f = self.sock.makefile("rwb")
        self.name = "%s:%d" % (host, port)

    def send(self, msg: Mapping[str, Any]) -> None:
        self.f.write((json.dumps(msg, ensure_ascii=False) + "\n").encode("utf-8"))
        self.f.flush()

    def recv(self) -> Dict[str, Any]:
        while True:
            line = self.f.readline()
            if not line:
                raise ConnectionError("게임이 연결을 닫음")
            if line.strip():
                return json.loads(line.decode("utf-8"))

    def request(self, msg: Mapping[str, Any], expect: str) -> Dict[str, Any]:
        self.send(msg)
        reply = self.recv()
        if reply.get("type") == P.ERROR:
            raise GameErrorReply(reply.get("code", "?"), reply.get("message", ""))
        if reply.get("type") != expect:
            raise AssertionError("%s 를 기대했으나 %r" % (expect, reply.get("type")))
        return reply

    def run_episode(
        self,
        cls: str,
        seed: int,
        params: Optional[Mapping[str, Any]] = None,
        scenario: str = "base",
        triggers: Optional[Tuple[str, Sequence[float]]] = None,
        on_trigger: Optional[Callable[[Dict[str, Any]], Mapping[str, Any]]] = None,
        initial_state: Optional[Mapping[str, Any]] = None,
    ) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
        """→ (result, 받은 trigger 메시지들)."""
        eid = "%s#check%d" % (self.name, next(_ids))
        msg: Dict[str, Any] = {"type": P.RUN_EPISODE, "episode_id": eid, "class": cls, "scenario": scenario,
                               "seed": int(seed), "params": dict(params or {})}
        if triggers:
            msg["triggers"] = {triggers[0]: list(triggers[1])}
        if initial_state is not None:
            msg["initial_state"] = dict(initial_state)
        self.send(msg)
        seen: List[Dict[str, Any]] = []
        while True:
            reply = self.recv()
            kind = reply.get("type")
            if kind == P.TRIGGER:
                seen.append(reply)
                params_out = dict(on_trigger(reply)) if on_trigger else {}
                self.send({"type": P.TRIGGER_RESPONSE, "episode_id": eid, "index": reply.get("index"), "params": params_out})
            elif kind == P.EPISODE_RESULT:
                return dict(reply.get("result") or {}), seen
            elif kind == P.ERROR:
                raise GameErrorReply(reply.get("code", "?"), reply.get("message", ""))
            else:
                raise AssertionError("run_episode 중 예상하지 못한 메시지 %r" % kind)

    def close(self) -> None:
        try:
            self.send({"type": P.BYE})
        except OSError:
            pass
        self.sock.close()


@dataclass
class CheckResult:
    name: str
    passed: bool
    detail: str = ""
    required: bool = True

    def line(self) -> str:
        mark = "PASS" if self.passed else ("FAIL" if self.required else "WARN")
        return "[%s] %s%s" % (mark, self.name, (" — " + self.detail) if self.detail else "")


def _minor(version: str) -> int:
    try:
        return int(str(version).split(".")[1])
    except (IndexError, ValueError):
        return 0


def description_problems(d: Mapping[str, Any]) -> List[str]:
    """describe 형식 + 학습 대상 규칙(블록 가중치만) 검사."""
    out: List[str] = []
    for key in ("protocol_version", "game", "time_limit", "max_deaths", "classes", "scenarios", "triggers"):
        if key not in d:
            out.append("description 필수 필드 누락: %s" % key)
    if out:
        return out
    if not d["classes"]:
        out.append("classes 비어 있음")
    if "base" not in [s["name"] if isinstance(s, dict) else s for s in d["scenarios"]]:
        out.append("scenario 'base' 필수")
    if not d["triggers"].get("state_features"):
        out.append("triggers.state_features 비어 있음")
    for c in d["classes"]:
        params = {p["name"]: p for p in c.get("params", [])}
        for p in params.values():
            low, high = p["range"]
            if not (low < high and low <= p["default"] <= high):
                out.append("%s.%s 범위/기본값 오류" % (c["name"], p["name"]))
        if "blocks" not in c:
            out.append("%s: classes[].blocks 없음 (v1.3)" % c["name"])
            continue
        weights = [b.get("weight") for b in c["blocks"]]
        if not weights:
            out.append("%s: 블록 가중치 없음" % c["name"])
        for w in weights:
            p = params.get(w)
            if p is None:
                out.append("%s: 가중치 %r 가 params 에 없음" % (c["name"], w))
            elif p.get("group", "bt") != "bt" or p["range"][0] < 0:
                out.append("%s.%s: 가중치는 group bt, 범위 0 이상" % (c["name"], w))
        wrong = [n for n, p in params.items() if bool(p.get("optimize", p.get("group", "bt") == "bt")) != (n in weights)]
        if wrong:
            out.append("%s: optimize 는 블록 가중치에만 true 여야 함: %s" % (c["name"], wrong))
    feats = d["triggers"].get("state_features") or []
    for s in d.get("initial_state") or []:
        lo, hi = s.get("range", [None, None])
        if s.get("name") not in feats:
            out.append("initial_state.%s: triggers.state_features 에 없음" % s.get("name"))
        if not (isinstance(lo, (int, float)) and isinstance(hi, (int, float)) and 0 <= lo < hi <= 1 and lo <= s.get("default", -1) <= hi):
            out.append("initial_state.%s: range [0,1] 안·default 는 range 안" % s.get("name"))
    return out


class Conformance:
    def __init__(self, host: str, port: int, timeout: float = 60.0) -> None:
        self.host, self.port, self.timeout = host, port, timeout
        self.results: List[CheckResult] = []
        self.c: Optional[ProbeClient] = None
        self.d: Dict[str, Any] = {}

    def _check(self, name: str, fn: Callable[[], str], required: bool = True) -> bool:
        try:
            detail = fn() or ""
            self.results.append(CheckResult(name, True, detail, required))
            return True
        except Exception as exc:  # 검사 실패 기록
            self.results.append(CheckResult(name, False, "%s: %s" % (type(exc).__name__, exc), required))
            return False

    def _expect_error(self, code: str, **kw: Any) -> str:
        assert self.c is not None
        try:
            self.c.run_episode(**kw)
        except GameErrorReply as exc:
            if exc.code != code:
                raise AssertionError("error code %r 기대, %r 받음" % (code, exc.code))
            return "error(%s)" % code
        raise AssertionError("error(%s) 를 기대했으나 정상 결과" % code)

    def run(self) -> List[CheckResult]:
        def hello() -> str:
            self.c = ProbeClient(self.host, self.port, self.timeout)
            r = self.c.request({"type": P.HELLO, "protocol_version": P.PROTOCOL_VERSION, "client": "btqa_game.conformance"}, P.HELLO_OK)
            return "protocol %s" % r.get("protocol_version")

        if not self._check("connect + hello", hello):
            return self.results
        c = self.c
        assert c is not None

        def describe() -> str:
            self.d = c.request({"type": P.DESCRIBE}, P.DESCRIPTION)
            fmt = [p for p in description_problems(self.d) if "blocks" not in p and "가중치" not in p and "optimize" not in p]
            assert not fmt, "; ".join(fmt)
            return "game=%s %s, classes=%s" % (self.d["game"], self.d.get("game_version"), [x["name"] for x in self.d["classes"]])

        if not self._check("describe 형식", describe):
            return self.results
        d = self.d

        def blocks() -> str:
            rule = [p for p in description_problems(d) if "blocks" in p or "가중치" in p or "optimize" in p]
            assert not rule, "; ".join(rule)
            return ", ".join("%s=%s" % (x["name"], [b["weight"] for b in x["blocks"]]) for x in d["classes"])

        self._check("블록 가중치 선언 (v1.3, 학습 대상 = 가중치뿐)", blocks, required=_minor(d["protocol_version"]) >= 3)

        cdesc = d["classes"][0]
        cls = cdesc["name"]
        defaults = {p["name"]: p["default"] for p in cdesc["params"]}
        trig = d["triggers"]
        ttype = trig.get("type", "boss_hp_below")
        thresholds = sorted((float(t) for t in trig.get("supported", [])), reverse=True) or [0.5]
        time_limit = float(d["time_limit"])
        first: List[Dict[str, Any]] = []

        def static_episode() -> str:
            r, _ = c.run_episode(cls, CHECK_SEED)
            first.append(r)
            assert r.get("seed") == CHECK_SEED, "result.seed 가 요청 seed 와 다름"
            assert 0 < float(r["elapsed"]) <= time_limit + 1e-6, "elapsed 범위 이상: %s" % r["elapsed"]
            assert 0.0 <= float(r["boss_hp_remaining"]) <= 1.0, "boss_hp_remaining 는 0~1"
            assert (r.get("clear_time") is not None) == bool(r["cleared"]), "cleared 와 clear_time 불일치"
            assert int(r.get("triggers_fired", 0)) == 0, "트리거를 요청하지 않았는데 triggers_fired=%s" % r.get("triggers_fired")
            return "cleared=%s elapsed=%.1f deaths=%s" % (r["cleared"], float(r["elapsed"]), r.get("deaths"))

        self._check("run_episode (기본 파라미터)", static_episode)

        def full_params() -> str:
            r, _ = c.run_episode(cls, CHECK_SEED, defaults)
            if first:
                assert r == first[0], "정의서 default 를 명시해도 결과가 같아야 함"
            return "ok"

        self._check("params 생략 = 정의서 default", full_params)

        def determinism() -> str:
            a, _ = c.run_episode(cls, CHECK_SEED + 1, defaults)
            b, _ = c.run_episode(cls, CHECK_SEED + 1, defaults)
            assert a == b, "같은 seed·params 인데 결과가 다름"
            return "동일"

        self._check("시드 결정성", determinism)

        def seed_variation() -> str:
            times = {c.run_episode(cls, CHECK_SEED + 10 + i, defaults)[0]["elapsed"] for i in range(4)}
            assert len(times) > 1, "seed 가 달라도 결과가 모두 같음 (난수가 seed 를 안 쓰는지 확인)"
            return "%d가지" % len(times)

        self._check("seed 에 따른 변동", seed_variation, required=False)

        weights = [b["weight"] for b in cdesc.get("blocks", [])]
        reply = {weights[0]: defaults[weights[0]]} if weights else {}

        def triggers() -> str:
            r, seen = c.run_episode(cls, CHECK_SEED, triggers=(ttype, thresholds), on_trigger=lambda m: reply)
            idx = [m.get("index") for m in seen]
            assert idx and idx[0] == 0, "첫 트리거는 index 0 (판 시작) 이어야 함: %s" % idx
            assert idx == list(range(len(idx))), "index 는 0,1,2,... 연속: %s" % idx
            assert len(idx) <= len(thresholds) + 1, "트리거 횟수 초과: %s" % idx
            assert int(r.get("triggers_fired", -1)) == len(idx), "result.triggers_fired != 받은 트리거 수"
            for m in seen:
                state = m.get("state") or {}
                missing = [f for f in trig["state_features"] if f not in state]
                assert not missing, "trigger.state 에 %s 없음" % missing
                bad = {k: v for k, v in state.items() if not (isinstance(v, (int, float)) and math.isfinite(v))}
                assert not bad, "state 값이 숫자가 아님: %s" % bad
                miss_t = [f for f in trig.get("totals", []) if f not in (m.get("totals") or {})]
                assert not miss_t, "trigger.totals 에 %s 없음" % miss_t
            if r["cleared"]:
                assert len(idx) == len(thresholds) + 1, "클리어했는데 모든 경계 트리거를 받지 못함: %s" % idx
            return "index=%s" % idx

        self._check("결정 트리거 왕복", triggers)

        def totals() -> str:
            assert all(k in trig.get("totals", []) for k in REQUIRED_TOTALS), "triggers.totals 에 %s 필요 (v1.1)" % (REQUIRED_TOTALS,)
            r, seen = c.run_episode(cls, CHECK_SEED + 2, triggers=(ttype, thresholds))
            el = [float(m["totals"]["elapsed"]) for m in seen]
            assert el[0] <= 1e-6, "index 0 의 elapsed 는 0 이어야 함"
            assert el == sorted(el), "elapsed 가 단조 증가하지 않음: %s" % el
            assert el[-1] <= float(r["elapsed"]) + 1e-6, "마지막 트리거 elapsed > result.elapsed"
            for key in ("deaths", "consumables_used"):
                seq = [float(m["totals"][key]) for m in seen] + [float(r.get(key, 0))]
                assert seq == sorted(seq), "%s 누적값이 감소함: %s" % (key, seq)
            return "elapsed=%s" % el

        self._check("trigger.totals 누적값 (v1.1, IMPALA 용)", totals, required=False)

        def stalled_field() -> str:
            r, _ = c.run_episode(cls, CHECK_SEED + 3)
            assert "stalled" in r, "result.stalled 없음 (v1.2)"
            assert isinstance(r["stalled"], bool), "result.stalled 는 bool"
            if r["stalled"]:
                assert not r.get("cleared") and not r.get("invalid"), "stalled 판은 cleared=false, invalid=false"
            return "stalled=%s" % r["stalled"]

        self._check("result.stalled 필드 (v1.2, 멈춘 판 = 실패)", stalled_field, required=False)

        p0 = cdesc["params"][0]
        span = abs(p0["range"][1] - p0["range"][0])
        self._check("unknown_param 오류", lambda: self._expect_error(P.E_UNKNOWN_PARAM, cls=cls, seed=1, params={"__no_such_param__": 1.0}))
        self._check("param_out_of_range 오류", lambda: self._expect_error(
            P.E_PARAM_OUT_OF_RANGE, cls=cls, seed=1, params={p0["name"]: p0["range"][1] + span + 1}))
        self._check("unknown_class 오류", lambda: self._expect_error(P.E_UNKNOWN_CLASS, cls="__no_such_class__", seed=1))
        self._check("unknown_scenario 오류", lambda: self._expect_error(P.E_UNKNOWN_SCENARIO, cls=cls, seed=1, scenario="__no_such_scenario__"))
        self._check("오류 후 연결 유지", lambda: (c.run_episode(cls, 2), "오류 후에도 같은 연결로 계속 사용 가능")[1])

        init = d.get("initial_state") or []
        if init:
            full = {s["name"]: s["default"] for s in init}
            low = {s["name"]: s["range"][0] for s in init}

            def init_default() -> str:
                a, _ = c.run_episode(cls, CHECK_SEED, defaults)
                b, _ = c.run_episode(cls, CHECK_SEED, defaults, initial_state=full)
                assert a == b, "initial_state = 선언 default 이면 생략한 판과 같은 결과여야 함"
                return "ok"

            def init_determinism() -> str:
                a, _ = c.run_episode(cls, CHECK_SEED + 4, defaults, initial_state=low)
                b, _ = c.run_episode(cls, CHECK_SEED + 4, defaults, initial_state=low)
                assert a == b, "같은 seed·params·initial_state 인데 결과가 다름"
                return "동일"

            def init_applied() -> str:
                _, seen = c.run_episode(cls, CHECK_SEED + 5, initial_state=low, triggers=(ttype, thresholds))
                st = seen[0].get("state") or {}
                off = {k: (st.get(k), v) for k, v in low.items() if st.get(k) is None or abs(float(st[k]) - v) > 0.02}
                assert not off, "trigger index 0 state 가 initial_state 와 다름 (state, 요청): %s" % off
                return "index 0 state = %s" % {k: st.get(k) for k in low}

            def init_errors() -> str:
                self._expect_error(P.E_UNKNOWN_PARAM, cls=cls, seed=1, initial_state={"__no_such_state__": 0.5})
                s0 = init[0]
                self._expect_error(P.E_PARAM_OUT_OF_RANGE, cls=cls, seed=1, initial_state={s0["name"]: s0["range"][1] + 1.0})
                self._expect_error(P.E_BAD_REQUEST, cls=cls, seed=1, initial_state={s0["name"]: "full"})
                return "unknown_param / param_out_of_range / bad_request"

            self._check("initial_state = default → 생략과 같은 결과 (v1.4)", init_default)
            self._check("initial_state 결정성 (v1.4)", init_determinism)
            self._check("initial_state 가 트리거 0 state 에 반영 (v1.4)", init_applied, required=False)
            self._check("initial_state 오류 코드 (v1.4)", init_errors)
        else:
            def init_refused() -> str:
                return self._expect_error(P.E_UNKNOWN_PARAM, cls=cls, seed=1, initial_state={"hp_ratio": 0.5})

            self._check("initial_state 미선언 → 요청 거부 unknown_param (v1.4)", init_refused, required=_minor(d["protocol_version"]) >= 4)

        def scenarios() -> str:
            names = [s["name"] if isinstance(s, dict) else s for s in d["scenarios"]]
            for sc in names:
                c.run_episode(cls, 3, scenario=sc)
            return "%d개" % len(names)

        self._check("모든 scenario 실행", scenarios)
        c.close()
        return self.results

    @property
    def passed(self) -> bool:
        return all(r.passed for r in self.results if r.required)


def check(host: str, port: int, timeout: float = 60.0, verbose: bool = True) -> bool:
    conf = Conformance(host, port, timeout)
    for r in conf.run():
        if verbose:
            print(r.line(), flush=True)
    if verbose:
        print("\n결과: %s" % ("PASS — 학습 플랫폼에 연결 가능" if conf.passed else "FAIL"))
    return conf.passed
