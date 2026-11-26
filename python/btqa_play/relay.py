"""PilotRelay — 대화형 에이전트(예: Claude Code 세션)가 직접 플레이어가 되는 중계기. LLM API 호출 없음.

LlmPilot 은 LLM API 를 스스로 부르는 루프다. 이 중계기는 반대로 루프를 바깥 에이전트에게 맡긴다:

    host  : 게임에 접속(hello·describe·prepare·reset)해 연결을 붙잡고, 127.0.0.1:<port> 에서 도구 호출을 기다린다
    step  : 도구 호출 1개 {"tool": ..., "input": {...}, "thought": "...", "kind": "thought"} 를 host 에 보내고
            결과(JSON 한 줄)를 출력하고 끝난다. thought 가 있으면 행동 전에 pilot_note(v0.3)로 게임 화면에 띄운다.
            tool = "note" 면 행동 없이 메모만, "intro" 면 처음 관찰·도구·게임 안내.

도구 = LlmPilot 과 같은 GameTools (전투는 학습된 BT·응답기가 끝낸다). finish 를 받으면 host 가
<out>/relay-<시각>/run.json·play_log.json 을 남기고 종료한다. 중계 메시지는 게임 프로토콜과 별개 (JSON Lines).

단계 형식 기록 (stateless=True, record=<jsonl>) — 증류용 교사 데이터:
    교사(대화형 에이전트)는 **학생과 같은 입력만** 보고 판단한다. intro·view·매 행동의 응답에 다음 단계 입력
    (actions.step_input = 지금 관찰 + 직전 행동 결과) 이 실리고, 교사는 그것만 읽고 다음 행동을 정한다.
    thought(--think) 가 학습 목표의 thought 가 된다. 행동마다 StepRecorder 로 JSONL 한 줄.

교정 데이터 (DAgger, student=LlmBackend): 매 단계 학생 모델의 제안(student)을 view·응답에 같이 싣는다. 교사가 정답을 보내며
    execute="student" 를 주면 **정답은 기록하고 게임에는 학생 행동을 실행**한다 → 학생이 실제로 빠지는 상태에서 정답이 쌓인다.
    execute 가 없으면(기본) 교사 행동을 실행 (학생이 헤매는 곳에서 끌어내기).

라벨 검토 (review.py): 정답으로 기록한 행동을 그대로 실행했는데 실패했거나(label_failed), 같은 행동이 진척 없이
    또 성공하면(repeat_no_progress) 응답에 "warning" 을 싣고 기록에 needs_review 를 남긴다. 교사가 확신하지 못하면
    step --unsure → 기록 uncertain=true (생각 문장에 글자로 넣지 않는다). 해소는 python -m btqa_play review / resolve.

확장 지점 (하위 클래스):
    prepare(client, desc) -> responder   로드아웃(set_loadout)·전투 응답기 준비. 기본 = 없음 (게임 default BT)
    run_meta() -> dict                   run.json 에 덧붙일 정보 (로드아웃 출처 등)
"""
from __future__ import annotations

import json
import socket
import time
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Union

from btqa_play.actions import ActionHistory, action_system_prompt, last_action_record, step_input, with_goal
from btqa_play.client import PlayClient, TriggerResponder
from btqa_play.prompt import build_system_prompt, default_goal
from btqa_play.branch import NON_ACTING_TOOLS
from btqa_play.pilot import progress_signature
from btqa_play.record import StepRecorder
from btqa_play.review import label_flags
from btqa_play.spec import PlaySpec, as_spec, effective_observation_drop, effective_progress_keys
from btqa_play.tools import GameTools, build_tools, compact_observation, to_text, tool_sources

DEFAULT_RELAY_PORT = 7899


class PilotRelay:
    def __init__(self, game: str, seed: int = 1, cls: Optional[str] = None, port: int = DEFAULT_RELAY_PORT,
                 out: str = "reports", pilot: str = "claude-code",
                 spec: Union[None, PlaySpec, Mapping[str, Any], str] = None,
                 stateless: bool = False, record: Optional[Union[str, Path]] = None, student: Any = None,
                 options: Optional[Mapping[str, Any]] = None, history: int = 0) -> None:
        self.game, self.seed, self.cls, self.port, self.out, self.pilot = game, seed, cls, port, Path(out), pilot
        self.options = dict(options or {})   # reset.options (게임 선언 시작 조건, 4.4.1)
        self.spec = as_spec(spec)   # 외부 플레이 스펙 (게임이 선언하지 않은 설명·스키마·안내)
        self.stateless = stateless or record is not None
        self.record_path = Path(record) if record else None
        self.recorder: Optional[StepRecorder] = None
        self.student = student       # 교정 데이터: 학생 LlmBackend (JsonActionBackend 등) — 제안만, 실행은 execute 로 고른다
        self._base_system = ""       # 학생 백엔드에 넘길 system (도구·형식은 백엔드가 붙인다)
        self._tools: List[Dict[str, Any]] = []
        self._proposal: Optional[Dict[str, Any]] = None
        self._system = ""            # 단계 형식 system (학생이 받는 글자)
        self._drop: List[str] = []
        self._last: Optional[Dict[str, Any]] = None
        self._hist = ActionHistory(history)   # 단계 입력의 recent_actions (학생 모델과 같은 K)
        self._obs: Dict[str, Any] = {}
        self.transcript: List[Dict[str, Any]] = []
        self.client: Any = None
        self._keys: Optional[List[str]] = None      # 진척 서명 키 (게임 → 스펙 progress_keys)
        self._prev: Optional[tuple] = None          # (직전 실행 행동, 성공 여부) — 반복 표시
        self.warnings = 0

    # ---------------------------------------------------------------- 확장 지점
    def prepare(self, client: PlayClient, desc: Mapping[str, Any]) -> Optional[TriggerResponder]:
        """reset 전에 한 번. 로드아웃을 보내고 전투 응답기를 돌려준다. 문제가 있으면 SystemExit."""
        return None

    def run_meta(self) -> Dict[str, Any]:
        return {}

    def notes(self) -> List[str]:
        return []

    def reset_options(self) -> Dict[str, Any]:
        """reset.options (게임이 선언한 시작 조건, PLAY_PROTOCOL 4.4.1). 기본 = 없음."""
        return dict(self.options)

    # ---------------------------------------------------------------- 실행
    def _view(self) -> Dict[str, Any]:
        """단계 형식: 지금 단계 입력 (학생이 받는 user 메시지 그대로) + 교정 모드면 학생 제안."""
        out: Dict[str, Any] = {"step_input": step_input(compact_observation(self._obs, self._drop), self._last,
                                                        self._hist.recent())}
        if self.student is not None:
            if self._proposal is None or self._proposal.get("_for") != out["step_input"]:
                self._proposal = dict(self._propose(out["step_input"]), _for=out["step_input"])
            out["student"] = {k: v for k, v in self._proposal.items() if k != "_for"}
        return out

    def _student_extra(self) -> Optional[Dict[str, Any]]:
        """교정 모드: 이 단계 학생 제안을 기록에 같이 남긴다 (정답과 다르면 DPO 의 rejected)."""
        if self.student is None or not self._proposal:
            return None
        return {"student": {k: v for k, v in self._proposal.items() if k != "_for"}}

    def _propose(self, user: str) -> Dict[str, Any]:
        self.student.start(self._base_system, self._tools, user)
        t = self.student.complete()
        c = t.calls[0] if t.calls else None
        if c is None or c.error:
            return {"error": (c.error if c else "도구 호출 없음"), "thought": "\n".join(t.texts)}
        return {"thought": "\n".join(t.texts), "tool": c.name, "input": dict(c.input)}

    def intro(self, desc: Mapping[str, Any], first: Mapping[str, Any]) -> Dict[str, Any]:
        """대화형 에이전트가 처음 읽을 것 = LlmPilot 이 LLM 에게 주는 것과 같은 정보."""
        if self.stateless:
            return dict(self._view(), system=self._system, tool_sources=tool_sources(desc, self.spec),
                        record=str(self.record_path) if self.record_path else None)
        return {"game": "%s %s" % (desc.get("game"), desc.get("game_version")),
                "goal": default_goal(desc, self.spec),
                "system_prompt": build_system_prompt(desc, self.spec),   # LlmPilot 의 LLM 이 받는 규칙 그대로
                "quests_in_game": desc.get("quests"),
                "tools": [{"name": t["name"], "description": t["description"], "args": t["input_schema"]["properties"]}
                          for t in build_tools(desc, self.spec, strict=False)],
                "tool_sources": tool_sources(desc, self.spec),
                "observation": compact_observation(first)}

    def serve(self) -> Path:
        t0 = time.time()
        with PlayClient.connect(self.game) as c:
            c.hello()
            desc = c.describe()
            responder = self.prepare(c, desc)
            first = c.reset(seed=self.seed, cls=self.cls, options=self.reset_options() or None)
            tools = GameTools(c, responder)
            if self.stateless:
                tl = build_tools(desc, self.spec)
                self._base_system = with_goal(build_system_prompt(desc, self.spec), default_goal(desc, self.spec))
                self._system = action_system_prompt(self._base_system, tl)
                self._tools = tl
                self._drop = effective_observation_drop(desc, self.spec)
                self._keys = effective_progress_keys(desc, self.spec)
                self._obs = dict(first)
                if self.record_path:
                    self.recorder = StepRecorder(self.record_path, run="relay-%s-s%s" % (time.strftime("%Y%m%d-%H%M%S"), self.seed),
                                                 source="teacher", model=self.pilot,
                                                 episode={"seed": self.seed, "class": self.cls, "options": self.reset_options()})
            intro = self.intro(desc, first)
            self.client = c
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as srv:
                srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                srv.bind(("127.0.0.1", self.port))
                srv.listen(1)
                print("relay ready 127.0.0.1:%d (game %s)" % (self.port, self.game), flush=True)
                while tools.finished is None:
                    conn, _ = srv.accept()
                    with conn:
                        line = conn.makefile("rb").readline().decode("utf-8", "replace").strip()
                        try:
                            req = json.loads(line)
                        except ValueError:
                            req = None
                        if not isinstance(req, dict):   # 빈 줄·포트 확인 등 잘못된 요청은 중계기를 죽이지 않는다
                            if line:
                                conn.sendall((to_text({"error": "bad request: JSON 객체 한 줄"}) + "\n").encode("utf-8"))
                            continue
                        out = self._handle(tools, req, intro)
                        conn.sendall((to_text(out) + "\n").encode("utf-8"))
            log = c.get_log()
            notes = c.notes + self.notes()
        return self._save(desc, tools, log, notes, round(time.time() - t0, 1))

    def _handle(self, tools: GameTools, req: Mapping[str, Any], intro: Mapping[str, Any]) -> Dict[str, Any]:
        name, inp = req.get("tool"), dict(req.get("input") or {})
        if name == "intro":   # 처음 관찰·도구 목록 다시 보기 (행동 아님)
            return dict(intro, **self._view()) if self.stateless else dict(intro)
        if name == "view" and self.stateless:   # 지금 단계 입력 다시 보기
            return self._view()
        thought, kind = req.get("thought"), str(req.get("kind") or "thought")
        shown = None
        if thought:
            shown = self.client.note(str(thought), kind)
        if name == "note":   # 메모만
            self.transcript.append({"note": thought, "kind": kind, "shown": shown})
            return {"ok": True, "shown": shown}
        # 교정 모드: 교사 정답(name·inp·thought)은 기록, 게임에는 학생 제안을 실행
        label = (str(name), dict(inp), thought)
        if self.stateless and req.get("execute") == "student" and self._proposal and self._proposal.get("tool"):
            name, inp = self._proposal["tool"], dict(self._proposal.get("input") or {})
        try:
            out = tools.execute(str(name), inp)
        except Exception as e:   # PlayError 등 → 세션 종료로 기록
            tools.finished = {"success": False, "summary": "relay error", "blocked": "game_error: %s" % e}
            out = {"error": str(e)}
        self.transcript.append({"tool": name, "input": inp, "thought": thought, "kind": kind if thought else None, "shown": shown,
                                "status": out.get("status"), "reason": out.get("reason"),
                                "error": out.get("error"), "combats": out.get("combats")})
        if not self.stateless:
            return out
        # 교사가 이 행동을 정할 때 본 입력 (아직 갱신 전 상태) → 기록 → 상태 갱신 → 다음 단계 입력을 돌려준다
        user_before = self._view()["step_input"]
        view_before = compact_observation(self._obs, self._drop)
        sig_before = (progress_signature(self._obs, self._keys), progress_signature(self._obs, None))
        self._last = last_action_record(str(name), inp, out)
        self._hist.push(self._last)
        if tools.last_observation:
            self._obs = dict(tools.last_observation)
        executed = (str(name), inp)
        ok = not self._last.get("error") and self._last.get("status") != "failed"
        flags = [] if str(name) in NON_ACTING_TOOLS else label_flags((label[0], label[1]), executed, self._last,
                            prev_executed=self._prev[0] if self._prev else None, prev_ok=bool(self._prev and self._prev[1]),
                            # 진척 = 진척 키가 바뀌었거나 관찰 상태(창·대화 글 등)가 바뀜 — 대화 넘기기 같은 같은 버튼 반복은 반복이 아니다
                            progressed=(progress_signature(self._obs, self._keys), progress_signature(self._obs, None)) != sig_before)
        self._prev = (executed, ok)
        grounding: List[str] = []
        if self.recorder is not None:
            res_for_label = self._last if (label[0], label[1]) == executed else {"executed": "student", "student_result": self._last}
            extra = dict(self._student_extra() or {})
            if flags:
                extra["needs_review"] = flags
            if req.get("uncertain"):
                extra["uncertain"] = True
            grounding = self.recorder.write(self._system, user_before, view_before, str(label[2] or ""), label[0], label[1],
                                            res_for_label, extra=extra or None)["grounding"]
        res: Dict[str, Any] = {"result": self._last, **self._view()}
        if grounding:
            res["grounding"] = grounding   # 이 행동은 입력에 근거가 없다 (학습 데이터에서 검토)
        if flags:
            self.warnings += 1
            res["warning"] = {"needs_review": flags,
                              "say": "정답으로 기록한 행동이 실패했거나 진척 없이 반복됐다 — 다음 단계는 교정하고, 끝나면 review 로 이 줄을 해소한다"}
        return res

    def _save(self, desc: Mapping[str, Any], tools: GameTools, log: Mapping[str, Any], notes: List[str], seconds: float) -> Path:
        out = self.out / ("relay-" + time.strftime("%Y%m%d-%H%M%S"))
        out.mkdir(parents=True, exist_ok=True)
        run = {"game": "%s %s" % (desc.get("game"), desc.get("game_version")), "seed": self.seed, "class": self.cls,
               "pilot": self.pilot, "notes": notes, "spec": self.spec.source or None,
               "mode": "stateless" if self.stateless else "conversation",
               "record": str(self.record_path) if self.record_path else None,
               "record_steps": self.recorder.step if self.recorder else 0,
               "record_flagged": self.recorder.flagged if self.recorder else 0,
               "record_needs_review": self.warnings,
               "end_reason": "finish", "finish": tools.finished, "actions": tools.actions,
               "combats": [x.summary() for x in tools.combats], "seconds": seconds,
               "last_observation": tools.last_observation, "transcript": self.transcript,
               "pilot_notes": self.client.pilot_notes, "pilot_notes_shown_in_game": self.client.supports_pilot_notes}
        run.update(self.run_meta())
        (out / "run.json").write_text(json.dumps(run, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
        (out / "play_log.json").write_text(json.dumps(log, ensure_ascii=False), encoding="utf-8")
        print("기록:", out, flush=True)
        return out


def relay_step(tool: str, tool_input: Mapping[str, Any], port: int = DEFAULT_RELAY_PORT,
               thought: Optional[str] = None, kind: str = "thought", execute: Optional[str] = None,
               uncertain: bool = False) -> str:
    req: Dict[str, Any] = {"tool": tool, "input": dict(tool_input)}
    if execute:
        req["execute"] = execute   # "student" = 교정 모드에서 학생 제안을 실행 (교사 행동은 정답으로만 기록)
    if uncertain:
        req["uncertain"] = True    # 교사가 확신하지 못한 라벨 → 기록 필드 (review.py)
    if thought:
        req.update(thought=thought, kind=kind)
    with socket.create_connection(("127.0.0.1", port)) as s:
        s.sendall((json.dumps(req, ensure_ascii=False) + "\n").encode("utf-8"))
        return s.makefile("rb").readline().decode("utf-8").strip()


def relay_approve(n: int = 1, port: int = DEFAULT_RELAY_PORT) -> List[Dict[str, Any]]:
    """교정 모드: 학생 제안을 정답으로 승인 (교사 = 학생과 같은 행동, 생각 문장도 학생 것) 을 n 번. 각 응답(dict) 목록.

    한 번에 여러 개 승인하면 학생의 틀린 행동을 놓치기 쉽다 (n 은 확실한 구간에만). 학생 제안이 없거나 판이 끝나면 멈춘다.
    생각 문장의 게임 고유 접두(상황 읽기 등)는 학습 데이터를 만들 때 prep 훅이 입력에서 다시 만든다."""
    out: List[Dict[str, Any]] = []
    for _ in range(n):
        st = (json.loads(relay_step("view", {}, port=port)).get("student") or {})
        if not st.get("tool"):
            out.append({"error": "학생 제안 없음", "student": st})
            break
        r = json.loads(relay_step(st["tool"], st.get("input") or {}, port=port, thought=st.get("thought") or None, execute="student"))
        out.append(r)
        if "step_input" not in r or r.get("warning"):   # 승인한 행동이 실패·반복이면 멈춘다 (다음은 교사가 본다)
            break
    return out


#: rules(step_input_text, proposal) -> None(승인) | {"tool","input","thought","rule"}(교정) | {"defer": 이유}(교사에게) — 게임 쪽 함수
RulesFn = Any


def relay_auto(rules: RulesFn, max_steps: int = 50, port: int = DEFAULT_RELAY_PORT,
               max_same_rule: int = 3, on_step: Any = None) -> Dict[str, Any]:
    """교정(DAgger) 모드 자동 진행: 학생 제안을 게임 쪽 규칙이 보고 — 문제 없으면 승인, 규칙 위반이면 규칙의 정답을 실행.

    멈추는 경우 (그다음은 교사가 본다): relay 경고(needs_review: 승인·교정한 행동이 실패하거나 진척 없이 반복),
    학생 제안 없음·형식 오류, 같은 규칙이 연속 max_same_rule 번, 판 끝, max_steps.
    규칙은 **입력(step_input)만** 보고 판단해야 한다 (학생이 배울 수 있게). 규칙이 판단할 수 없는 장면은 None 이 아니라
    교사에게 맡기는 편이 낫다 — 규칙이 None 을 돌려주면 승인, {"defer": 이유} 면 아무것도 하지 않고 멈춘다 (교사 차례)."""
    log: List[Dict[str, Any]] = []
    same, last_rule, stop = 0, None, "max_steps"
    for _ in range(max_steps):
        view = json.loads(relay_step("view", {}, port=port))
        st = view.get("student") or {}
        if "step_input" not in view:
            stop = "판 끝"
            break
        if not st.get("tool"):
            stop = "학생 제안 없음: %s" % (st.get("error") or "")
            break
        fix = rules(view["step_input"], st)
        if fix and fix.get("defer"):
            stop = "교사 필요: %s" % fix["defer"]
            break
        if fix:
            same = same + 1 if fix.get("rule") == last_rule else 1
            last_rule = fix.get("rule")
            if same > max_same_rule:
                stop = "같은 규칙 %s 연속 %d번" % (last_rule, same - 1)
                break
            r = json.loads(relay_step(fix["tool"], fix.get("input") or {}, port=port, thought=fix.get("thought") or None))
            entry = {"kind": "corrected", "rule": fix.get("rule"), "student": st, "label": fix}
        else:
            same, last_rule = 0, None
            r = json.loads(relay_step(st["tool"], st.get("input") or {}, port=port, thought=st.get("thought") or None,
                                      execute="student"))
            entry = {"kind": "approved", "student": st}
        entry["result"] = r.get("result")
        log.append(entry)
        if on_step:
            on_step(entry)
        if r.get("warning"):
            stop = "경고 %s" % ",".join(r["warning"]["needs_review"])
            break
        if "step_input" not in r or (entry["kind"] == "approved" and st["tool"] == "finish") or \
                (entry["kind"] == "corrected" and fix["tool"] == "finish"):
            stop = "판 끝"
            break
    return {"stop": stop, "steps": len(log), "approved": sum(e["kind"] == "approved" for e in log),
            "corrected": sum(e["kind"] == "corrected" for e in log),
            "rules": {k: sum(1 for e in log if e.get("rule") == k) for k in {e.get("rule") for e in log if e.get("rule")}},
            "log": log}


__all__ = ["PilotRelay", "relay_step", "relay_approve", "relay_auto", "DEFAULT_RELAY_PORT"]
