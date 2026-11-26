"""평가 스위트 — SFT 이후 단계(DPO·RFT)를 판정할 수 있는 평가. 표준 라이브러리만.

왜 필요한가 (AI_QA 2026-11-25): 기준 모델(SFT+DAgger)이 평가 seed 1~5 를 모두 깨자 DPO·RFT 는 **오를 곳이 없었다**.
게다가 seed 가 튜토리얼 상황을 거의 바꾸지 않아(39단계 중 33개 같은 입력) 5판 = 사실상 1판이었다.
후속 학습의 이득은 "아는 판을 또 깨는 것"이 아니라 **흔들려도 깨는 것(강건성)** 과 **안 본 조건에서 깨는 것(일반화)** 이다.

스위트 파일 (btqa-eval-suite/1, 게임 쪽이 쓴다 — 조건 값은 게임 내용이고 형식만 SDK):
    {"format": "btqa-eval-suite/1", "name": "...",
     "defaults": {"max_actions": 70, "temperature": 0.0, "repeats": 1},
     "conditions": [
       {"name": "seen",    "group": "seen",   "seeds": "1-5"},                                  ← 학습에 쓴 조건 (회귀 확인)
       {"name": "robust",  "group": "robust", "seeds": "1-5", "temperature": 0.7, "repeats": 4},  ← 샘플링으로 흔들기
       {"name": "unseen",  "group": "unseen", "seeds": "101-110", "class": "c2",
        "options": {"start_gold": 0}, "prepare": {"loadout": "..."}},                         ← 안 본 시작 조건
       {"name": "rft",     "group": "train",  "seeds": "11-20"}],                              ← 학습용 (평가·비교에서 뺀다)
     "adopt": {"must_not_regress": ["seen"], "tolerance": 0.0, "improve_any": ["robust", "unseen"], "min_gain": 0.1,
               "metric": "progress"}}
    - options = reset.options (게임이 play_description.reset_options 로 선언한 키만, PLAY_PROTOCOL 4.4.1). 선언이 없으면 그 조건은
      건너뛰고 이유를 남긴다 (게임이 아직 지원하지 않는 조건을 미리 적어 둘 수 있다).
    - prepare = 판 준비 훅(게임 쪽, 예: 로드아웃)에 넘길 인자.
    - group "train" 은 학습 조건 (RFT·on-policy 쌍이 쓴다). 평가 시드와 학습 시드를 섞지 않는다.

지표 (판마다): success = 퀘스트 전부 완료, progress = 완료 퀘스트 / 전체 (PLAY_PROTOCOL 4.2 quests·4.8 quest_completed).
묶음마다: 판 수, success 비율 + Wilson 95% 구간, progress 평균 + 95% 구간, **다른 궤적 수** (seed 가 판을 실제로 바꾸는지).
"""
from __future__ import annotations

import hashlib
import json
import math
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Union

from btqa_play.branch import PrepareFn, call_prepare
from btqa_play.client import PlayClient
from btqa_play.evaluate import parse_seeds, row_of
from btqa_play.pilot import LlmPilot, PilotConfig
from btqa_play.record import StepRecorder
from btqa_play.spec import PlaySpec
from btqa_play.tools import compact_observation

SUITE_FORMAT = "btqa-eval-suite/1"
TRAIN_GROUP = "train"
COND_KEYS = ("name", "group", "seeds", "class", "options", "prepare", "temperature", "repeats", "max_actions", "max_calls", "note")
ADOPT_KEYS = ("must_not_regress", "tolerance", "improve_any", "min_gain", "metric")
METRICS = ("progress", "success")


# ------------------------------------------------------------------ 파일
def validate_suite(d: Mapping[str, Any]) -> List[str]:
    out: List[str] = []
    if d.get("format") != SUITE_FORMAT:
        out.append("format 은 %s" % SUITE_FORMAT)
    conds = d.get("conditions")
    if not isinstance(conds, list) or not conds:
        out.append("conditions: 조건 목록이 필요하다")
        conds = []
    names = set()
    for i, c in enumerate(conds):
        w = "conditions[%d]" % i
        if not isinstance(c, Mapping):
            out.append(w + ": 객체가 아님")
            continue
        unknown = sorted(set(c) - set(COND_KEYS))
        if unknown:
            out.append("%s: 모르는 키 %s" % (w, unknown))
        if not c.get("name") or not c.get("group"):
            out.append(w + ": name·group 필요")
        if c.get("name") in names:
            out.append("%s: name 중복 %r" % (w, c.get("name")))
        names.add(c.get("name"))
        try:
            if not parse_seeds(str(c.get("seeds", ""))):
                out.append(w + ": seeds 비었음")
        except ValueError:
            out.append(w + ": seeds 형식 (예 '1-5' / '1,3,7')")
        for k in ("options", "prepare"):
            if k in c and not isinstance(c[k], Mapping):
                out.append("%s.%s: 객체여야 한다" % (w, k))
        t = c.get("temperature", 0.0)
        if not isinstance(t, (int, float)) or t < 0:
            out.append(w + ".temperature ≥ 0")
    adopt = d.get("adopt") or {}
    unknown = sorted(set(adopt) - set(ADOPT_KEYS))
    if unknown:
        out.append("adopt: 모르는 키 %s" % unknown)
    if adopt.get("metric", "progress") not in METRICS:
        out.append("adopt.metric 은 %s" % "|".join(METRICS))
    groups = {c.get("group") for c in conds if isinstance(c, Mapping)}
    for k in ("must_not_regress", "improve_any"):
        for g in adopt.get(k) or []:
            if g not in groups:
                out.append("adopt.%s: 조건에 없는 group %r" % (k, g))
            if g == TRAIN_GROUP:
                out.append("adopt.%s: train 묶음으로 판정하지 않는다 (학습 조건)" % k)
    return out


def load_suite(path: Union[str, Path, Mapping[str, Any]]) -> Dict[str, Any]:
    d = dict(path) if isinstance(path, Mapping) else json.loads(Path(path).read_text(encoding="utf-8"))
    problems = validate_suite(d)
    if problems:
        raise ValueError("평가 스위트: " + "; ".join(problems))
    return d


def conditions(suite: Mapping[str, Any], groups: Optional[Sequence[str]] = None,
               include_train: bool = False) -> List[Dict[str, Any]]:
    """defaults 를 채운 조건 목록. groups 를 주면 그 묶음만. train 은 include_train 일 때만."""
    base = {"temperature": 0.0, "repeats": 1, "options": {}, "prepare": {}, "class": None}
    base.update(suite.get("defaults") or {})
    out = []
    for c in suite.get("conditions") or []:
        if c.get("group") == TRAIN_GROUP and not include_train and not (groups and TRAIN_GROUP in groups):
            continue
        if groups and c.get("group") not in groups:
            continue
        out.append(dict(base, **c))
    return out


# ------------------------------------------------------------------ 통계
def wilson(k: int, n: int, z: float = 1.96) -> List[float]:
    if n <= 0:
        return [0.0, 1.0]
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [round(max(0.0, c - h), 3), round(min(1.0, c + h), 3)]


def mean_ci(xs: Sequence[float], z: float = 1.96) -> Dict[str, Any]:
    n = len(xs)
    if n == 0:
        return {"mean": None, "ci": None}
    m = sum(xs) / n
    sd = math.sqrt(sum((x - m) ** 2 for x in xs) / (n - 1)) if n > 1 else 0.0
    h = z * sd / math.sqrt(n) if n > 1 else 0.0
    return {"mean": round(m, 3), "ci": [round(max(0.0, m - h), 3), round(min(1.0, m + h), 3)]}


def aggregate(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    n = len(rows)
    k = sum(1 for r in rows if r.get("success"))
    prog = mean_ci([float(r.get("progress") or 0.0) for r in rows])
    return {"runs": n, "success": round(k / n, 3) if n else None, "success_ci": wilson(k, n),
            "progress": prog["mean"], "progress_ci": prog["ci"],
            "distinct_trajectories": len({r.get("trajectory") for r in rows}),
            "distinct_start_states": len({r.get("start_state") for r in rows})}


def _hash(x: Any) -> str:
    return hashlib.sha1(json.dumps(x, sort_keys=True, ensure_ascii=False, default=str).encode("utf-8")).hexdigest()[:12]


#: 시작 상태 비교에서 빼는 관찰 필드 (seed 와 무관하게 바뀌는 것)
_VOLATILE = ("seq", "game_time", "events", "type")


def start_state_hash(obs: Mapping[str, Any]) -> str:
    return _hash({k: v for k, v in compact_observation(obs).items() if k not in _VOLATILE})


def trajectory_hash(run: Any) -> str:
    return _hash([(t.get("tool"), t.get("input")) for t in (run.transcript or []) if t.get("tool")])


# ------------------------------------------------------------------ 실행
def run_suite(client: PlayClient, backend: Any, suite: Mapping[str, Any], prepare: Optional[PrepareFn] = None,
              spec: Union[None, PlaySpec, Mapping[str, Any], str] = None, groups: Optional[Sequence[str]] = None,
              out_dir: Union[None, str, Path] = None, record: bool = False, goal: Optional[str] = None,
              on_run: Optional[Callable[[Dict[str, Any]], None]] = None, history: int = 0,
              **meta: Any) -> Dict[str, Any]:
    """client 는 hello·describe 를 마친 상태. history = 학생 입력의 recent_actions 개수 (학습 형식과 같게, 요약에 남긴다). backend.temperature 가 있으면 조건마다 바꾼다 (샘플링 강건성).
    결과 = {conditions: {이름: 묶음 통계 + rows}, groups: {묶음: 통계}, skipped, warnings}."""
    desc = client.description
    total = len(desc.get("quests") or [])
    declared = set(client.reset_option_names)
    out = Path(out_dir) if out_dir else None
    if out:
        out.mkdir(parents=True, exist_ok=True)
    steps = str(out / "steps.jsonl") if (out and record) else None
    cond_out: Dict[str, Any] = {}
    by_group: Dict[str, List[Dict[str, Any]]] = {}
    skipped: Dict[str, str] = {}
    warnings: List[str] = []
    base_temp = getattr(backend, "temperature", None)
    for c in conditions(suite, groups):
        missing = sorted(set(c.get("options") or {}) - declared)
        if missing:
            skipped[c["name"]] = "게임이 reset 옵션 %s 를 선언하지 않음 (PLAY_PROTOCOL 4.4.1)" % missing
            continue
        responder = call_prepare(prepare, client, desc, c.get("prepare") or None)
        temp = float(c.get("temperature") or 0.0)
        if hasattr(backend, "temperature"):
            backend.temperature = temp
        elif temp > 0:
            warnings.append("%s: 백엔드에 temperature 가 없어 탐욕으로 돌림" % c["name"])
        cfg = PilotConfig(max_actions=int(c.get("max_actions") or 150), max_calls=int(c.get("max_calls") or 250),
                          goal=goal, stateless=True, history=history)
        rows = []
        for seed in parse_seeds(str(c["seeds"])):
            for rep in range(int(c.get("repeats") or 1)):
                run_name = "eval-%s-%s-s%d-r%d" % (time.strftime("%Y%m%d-%H%M%S"), c["name"], seed, rep)
                rec = StepRecorder(steps, run=run_name, source="model", model=getattr(backend, "label", ""),
                                   episode={"seed": seed, "class": c.get("class"), "options": c.get("options")}) if steps else None
                obs = client.reset(seed=seed, cls=c.get("class"), options=c.get("options") or None)
                run = LlmPilot(client, responder=responder, config=cfg, backend=backend, spec=spec, recorder=rec).run(obs)
                row = row_of(seed, run, client.get_log())
                row.update(condition=c["name"], group=c["group"], repeat=rep, temperature=temp,
                           progress=round(row["quests_done"] / total, 3) if total else 0.0,
                           success=bool(total) and row["quests_done"] >= total,
                           start_state=start_state_hash(obs), trajectory=trajectory_hash(run))
                rows.append(row)
                if on_run:
                    on_run(dict(row, quests_total=total))
        agg = aggregate(rows)
        seeds_n = len(parse_seeds(str(c["seeds"])))
        if seeds_n > 1 and agg["distinct_start_states"] == 1:
            warnings.append("%s: seed %d개의 시작 상태가 모두 같다 — 게임이 seed 로 상황을 바꾸지 않으면 판 수가 표본 수가 아니다"
                            % (c["name"], seeds_n))
        if seeds_n > 1 and temp == 0 and agg["distinct_trajectories"] == 1:
            warnings.append("%s: 탐욕 실행 %d판의 궤적이 모두 같다 — 사실상 1판" % (c["name"], len(rows)))
        cond_out[c["name"]] = dict(agg, group=c["group"], temperature=temp, rows=rows)
        by_group.setdefault(c["group"], []).extend(rows)
    if hasattr(backend, "temperature"):
        backend.temperature = base_temp
    summary = dict(meta, history=history, format="btqa-eval-result/1", suite=suite.get("name"), game="%s %s" % (desc.get("game"), desc.get("game_version")),
                   llm=getattr(backend, "label", ""), quests_total=total,
                   groups={g: aggregate(rs) for g, rs in by_group.items()}, conditions=cond_out, skipped=skipped,
                   warnings=warnings, adopt=suite.get("adopt") or {}, record=steps)
    if out:
        (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    return summary


# ------------------------------------------------------------------ 채택 판정
def compare(base: Mapping[str, Any], cand: Mapping[str, Any], adopt: Optional[Mapping[str, Any]] = None) -> Dict[str, Any]:
    """기준 모델 결과 vs 후보 결과 (run_suite 요약 2개) → 채택 여부와 이유.

    규칙 (adopt, 없으면 스위트에 적힌 것):
      must_not_regress 묶음은 후보 ≥ 기준 − tolerance,
      improve_any 묶음 중 하나 이상에서 후보 − 기준 ≥ min_gain.
    둘 다 맞아야 채택. 판 수가 적어 구간이 겹치면 경고 (채택은 규칙대로, 판단은 사람이)."""
    rule = dict(base.get("adopt") or {})
    rule.update(adopt or {})
    metric = rule.get("metric", "progress")
    tol, gain = float(rule.get("tolerance", 0.0)), float(rule.get("min_gain", 0.05))
    bg, cg = base.get("groups") or {}, cand.get("groups") or {}
    reasons: List[str] = []
    warnings: List[str] = []
    table = {}
    for g in sorted(set(bg) | set(cg)):
        b, c = (bg.get(g) or {}).get(metric), (cg.get(g) or {}).get(metric)
        table[g] = {"base": b, "cand": c, "diff": None if b is None or c is None else round(c - b, 3),
                    "base_ci": (bg.get(g) or {}).get(metric + "_ci"), "cand_ci": (cg.get(g) or {}).get(metric + "_ci")}
    ok = True
    for g in rule.get("must_not_regress") or []:
        t = table.get(g)
        if not t or t["diff"] is None:
            ok = False
            reasons.append("%s: 결과 없음" % g)
        elif t["diff"] < -tol:
            ok = False
            reasons.append("%s 후퇴 %.3f → %.3f (허용 %.3f)" % (g, t["base"], t["cand"], tol))
    gains = []
    for g in rule.get("improve_any") or []:
        t = table.get(g)
        if not t or t["diff"] is None:
            warnings.append("%s: 결과 없음 (건너뛴 조건?)" % g)
            continue
        if t["base"] is not None and t["base"] >= 1.0:
            warnings.append("%s: 기준 모델이 이미 %s=1.0 — 오를 곳이 없다 (평가가 포화, 조건을 어렵게)" % (g, metric))
        if t["diff"] >= gain:
            gains.append(g)
            if t["base_ci"] and t["cand_ci"] and t["cand_ci"][0] <= t["base_ci"][1]:
                warnings.append("%s: 이득 %.3f 이지만 95%% 구간이 겹친다 — 판을 늘려 다시 확인" % (g, t["diff"]))
    if rule.get("improve_any") and not gains:
        ok = False
        reasons.append("improve_any %s 중 %.3f 이상 오른 묶음 없음" % (list(rule.get("improve_any")), gain))
    if ok:
        reasons.append("채택: %s 에서 이득, 회귀 없음" % gains if gains else "채택: 회귀 없음")
    return {"adopt": ok, "metric": metric, "rule": rule, "table": table, "reasons": reasons, "warnings": warnings,
            "base": base.get("llm"), "cand": cand.get("llm")}


__all__ = ["SUITE_FORMAT", "TRAIN_GROUP", "validate_suite", "load_suite", "conditions", "wilson", "mean_ci", "aggregate",
           "run_suite", "compare", "start_state_hash", "trajectory_hash"]


# ------------------------------------------------------------------ DAgger·학습 데이터 조건 검사 (DISTILL_SDK.md 5.1)
def data_checks(records: Sequence[Mapping[str, Any]], suite: Optional[Mapping[str, Any]] = None) -> Dict[str, Any]:
    """학습에 쓸 단계 기록이 DAgger 조건을 지키는지 (게임 무관).

    - 평가 누수: 기록의 판(seed·class·options)이 스위트 평가 조건(train 묶음 제외)과 같으면 그 판을 센다.
      예전 기록처럼 class 를 모르면 seed 만으로 겹침을 본다 (보수적으로 경고).
    - 시작 상태 다양성: 판마다 첫 단계 관찰(변하는 키 제외)의 해시 — 판이 여러 개인데 시작 상태가 하나면 seed 가 판을 바꾸지 않는다.
    - 판 시작 조건 수: 서로 다른 (seed, class, options).
    결과 {"runs", "episodes", "distinct_start_states", "eval_overlap_runs", "warnings"}."""
    from btqa_play.record import episode_of
    first: Dict[str, Mapping[str, Any]] = {}
    for r in records:
        run = str(r.get("run"))
        if run not in first or int(r.get("step") or 0) < int(first[run].get("step") or 0):
            first[run] = r
    starts, eps, overlap = set(), set(), []
    evals = []
    for c in conditions(suite or {}):
        evals.append((c.get("class"), json.dumps(c.get("options") or {}, sort_keys=True), set(parse_seeds(str(c["seeds"])))))
    for run, r in first.items():
        try:
            obs = json.loads(r["messages"][1]["content"]).get("observation") or {}
            starts.add(_hash({k: v for k, v in obs.items() if k not in _VOLATILE}))
        except (KeyError, IndexError, ValueError, TypeError, AttributeError):
            pass
        ep = episode_of(r)
        opts = json.dumps(ep.get("options") or {}, sort_keys=True)
        eps.add((ep.get("seed"), ep.get("class"), opts))
        if "seed" in ep and any(int(ep["seed"]) in seeds and (ep.get("class") is None or cls is None or ep.get("class") == cls)
                                and opts == copts for cls, copts, seeds in evals):
            overlap.append(run)
    warn = []
    if overlap:
        warn.append("평가 조건과 같은 판 %d개 — 평가가 학습에 샌다 (스위트 train 묶음으로 모은다)" % len(overlap))
    if len(first) > 1 and len(starts) <= 1:
        warn.append("판 %d개의 시작 상태가 모두 같다 — seed 가 판을 바꾸지 않는다 (게임에 seed 변화·reset.options 요청)" % len(first))
    elif len(first) >= 4 and len(starts) * 2 <= len(first):
        warn.append("시작 상태 %d가지 / 판 %d개 — 같은 판이 많다" % (len(starts), len(first)))
    return {"runs": len(first), "episodes": len(eps), "distinct_start_states": len(starts),
            "eval_overlap_runs": sorted(overlap), "warnings": warn}
