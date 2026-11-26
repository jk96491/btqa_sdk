"""같은 상태에서 후보 행동 여러 개 시험하기 — RFT 채점·on-policy 선호 쌍의 공통 부품. 표준 라이브러리만.

    br = Brancher("127.0.0.1:7802", prepare=hook)          # 시험용(scratch) 게임 인스턴스 1개
    out = br.try_action(Episode(seed=3, cls="warrior"), prefix, "move_to", {"target": "npc_a"})

상태 만들기 두 가지 (게임이 고른다, PLAY_PROTOCOL 4.11):
    snapshot  게임이 capabilities.snapshot 을 선언하면: 상태마다 한 번 재생 → snapshot → 후보마다 restore (빠르다)
    replay    선언이 없으면: 후보마다 reset(seed, class, options) + prefix 재생 (같은 seed + 같은 행동열 → 같은 상태, 4.4·4.8)

게임 내용 없음: 행동은 GameTools 가 그대로 보내고, 점수는 쓰는 쪽 reward 훅이 매긴다.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

from btqa_play.client import PlayClient
from btqa_play.tools import GameTools

Action = Tuple[str, Dict[str, Any]]
#: prepare(client, desc, **args) -> 전투 응답기 또는 None (로드아웃 등 판 시작 준비, 게임 쪽 함수)
PrepareFn = Callable[..., Any]


@dataclass
class Episode:
    """판 하나의 시작 조건 (reset 인자 + 판 준비 인자)."""
    seed: int = 1
    cls: Optional[str] = None
    options: Dict[str, Any] = field(default_factory=dict)      # reset.options (게임 선언 키만, 4.4.1)
    prepare: Dict[str, Any] = field(default_factory=dict)      # prepare 훅에 넘길 인자 (예: 로드아웃 경로)

    def key(self) -> str:
        import json
        return json.dumps([self.seed, self.cls, self.options, self.prepare], sort_keys=True, ensure_ascii=False)


def call_prepare(prepare: Optional[PrepareFn], client: PlayClient, desc: Mapping[str, Any],
                 args: Optional[Mapping[str, Any]] = None) -> Any:
    """prepare 훅 호출 — 인자가 없으면 예전 모양 prepare(client, desc) 그대로."""
    if prepare is None:
        return None
    return prepare(client, desc, **dict(args)) if args else prepare(client, desc)


class Brancher:
    """시험용 게임 연결 1개. 상태 = (Episode, prefix 행동열)."""

    def __init__(self, endpoint: str, prepare: Optional[PrepareFn] = None, use_snapshot: Optional[bool] = None,
                 client: Optional[PlayClient] = None) -> None:
        self.c = client or PlayClient.connect(endpoint)
        if not self.c.hello_msg:
            self.c.hello()
        self.desc = self.c.description or self.c.describe()
        self.prepare = prepare
        self.use_snapshot = self.c.supports_snapshot if use_snapshot is None else use_snapshot
        self._prepared: Optional[str] = None     # 지금 로드아웃을 준비한 Episode.prepare
        self.responder: Any = None
        self._snap: Optional[Tuple[str, int, str]] = None   # (episode key, prefix 길이, snapshot id)
        self._snap_prefix: List[Action] = []
        self.replays = 0                          # prefix 재생 횟수 (비용 확인용)
        self.state_obs: Dict[str, Any] = {}       # 마지막으로 만든 상태의 관찰 (보상 훅의 obs_before)

    def _start(self, ep: Episode) -> GameTools:
        import json
        pk = json.dumps(ep.prepare, sort_keys=True)
        if self._prepared != pk:
            self.responder = call_prepare(self.prepare, self.c, self.desc, ep.prepare)
            self._prepared = pk
        self.state_obs = dict(self.c.reset(seed=ep.seed, cls=ep.cls, options=ep.options or None))
        return GameTools(self.c, self.responder)

    def _replay(self, ep: Episode, prefix: Sequence[Action]) -> GameTools:
        tools = self._start(ep)
        for t, i in prefix:
            tools.execute(t, dict(i))
            if tools.last_observation:
                self.state_obs = dict(tools.last_observation)
        self.replays += 1
        return tools

    def _at(self, ep: Episode, prefix: Sequence[Action]) -> GameTools:
        """prefix 끝 상태의 GameTools."""
        if not self.use_snapshot:
            return self._replay(ep, prefix)
        if self._snap and self._snap[0] == ep.key() and self._snap[1] == len(prefix) and self._snap_prefix == list(prefix):
            self.state_obs = dict(self.c.restore(self._snap[2]))
            return GameTools(self.c, self.responder)
        tools = self._replay(ep, prefix)
        self._snap = (ep.key(), len(prefix), self.c.snapshot())
        self._snap_prefix = [(t, dict(i)) for t, i in prefix]
        return tools

    def observation_at(self, ep: Episode, prefix: Sequence[Action]) -> Dict[str, Any]:
        """prefix 상태의 관찰 (재생 확인·보상 훅의 obs_before)."""
        self._at(ep, prefix)
        return dict(self.state_obs)

    def try_action(self, ep: Episode, prefix: Sequence[Action], tool: str, inp: Mapping[str, Any]) -> Dict[str, Any]:
        """prefix 상태에서 행동 하나를 실행한 결과 (GameTools.execute 출력, 오류는 {"error"})."""
        try:
            tools = self._at(ep, prefix)
            return tools.execute(tool, dict(inp))
        except Exception as e:   # 전투 중 오류·연결 문제 → 후보 실패로
            return {"error": str(e)}

    def try_actions(self, ep: Episode, prefix: Sequence[Action],
                    candidates: Sequence[Action]) -> List[Dict[str, Any]]:
        """같은 상태에서 후보 여러 개. 똑같은 행동은 한 번만 실행 (게임 결정성)."""
        import json
        cache: Dict[str, Dict[str, Any]] = {}
        out = []
        for t, i in candidates:
            k = json.dumps([t, {a: b for a, b in i.items() if b is not None}], sort_keys=True, ensure_ascii=False)
            if k not in cache:
                cache[k] = self.try_action(ep, prefix, t, i)
            out.append(cache[k])
        return out

    def close(self) -> None:
        self.c.close()


#: 게임 상태를 바꾸지 않는 도구 (재생에서 뺀다)
NON_ACTING_TOOLS = ("finish", "inspect_ui")


def executed_action(rec: Mapping[str, Any]) -> Optional[Action]:
    """단계 기록 한 줄에서 게임에 실제로 실행된 행동. 상태를 바꾸지 않는 도구면 None.

    result 는 last_action_record (실행된 tool·input). 교정(DAgger) 모드에서 학생 행동을 실행했으면
    result = {"executed": "student", "student_result": ...} (relay.py)."""
    res = rec.get("result") or {}
    src = res.get("student_result") if res.get("executed") == "student" else res
    if not isinstance(src, Mapping) or not src.get("tool"):
        src = rec.get("target") or {}
    if src.get("tool") in (None, "") or src.get("tool") in NON_ACTING_TOOLS:
        return None
    return str(src["tool"]), dict(src.get("input") or {})


def prefixes(records: Sequence[Mapping[str, Any]]) -> List[List[Action]]:
    """한 판의 단계 기록(step 순) → 단계마다 그 단계 **직전까지** 실행된 행동열 (그 단계 입력의 상태를 만드는 prefix)."""
    out: List[List[Action]] = []
    done: List[Action] = []
    for r in records:
        out.append(list(done))
        a = executed_action(r)
        if a is not None:
            done.append(a)
    return out


__all__ = ["Brancher", "Episode", "PrepareFn", "Action", "NON_ACTING_TOOLS", "call_prepare", "executed_action", "prefixes"]
