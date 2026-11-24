"""여러 seed 평가 — 같은 LLM 으로 새 게임을 seed 마다 플레이하고 요약한다 (기준선·증류 전후·모델 비교).

    summary = run_batch(client, backend, seeds=[1, 2, 3], config=PilotConfig(stateless=True), spec=..., responder=...,
                        out_dir="reports/batch-…", record=True)

- client 는 hello·describe(·set_loadout)를 마친 상태. 판마다 reset(seed) 로 새 게임 (PLAY_PROTOCOL 4.4).
- 판별 PilotRun → run-s<seed>.json, 단계 기록(record=True, stateless) → steps.jsonl (record.py 형식), 요약 → summary.json
- 게임 고유 판단 없음: "완료한 퀘스트" = play_log 의 quest_completed 사건, 전체 = play_description.quests 수 (프로토콜 4.2·4.8)
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Union

from btqa_play.client import PlayClient
from btqa_play.llm.base import LlmBackend
from btqa_play.pilot import LlmPilot, PilotConfig, PilotRun
from btqa_play.record import StepRecorder
from btqa_play.spec import PlaySpec


def parse_seeds(text: str) -> List[int]:
    """"1-5" / "1,3,7" / "1-3,10" → [..]."""
    out: List[int] = []
    for part in str(text).split(","):
        part = part.strip()
        if "-" in part:
            lo, hi = part.split("-", 1)
            out += list(range(int(lo), int(hi) + 1))
        elif part:
            out.append(int(part))
    return out


def quests_completed(log: Mapping[str, Any]) -> List[str]:
    return [e[2].get("quest") for e in log.get("entries", []) if len(e) > 2 and e[1] == "quest_completed"]


def row_of(seed: int, run: PilotRun, log: Mapping[str, Any]) -> Dict[str, Any]:
    done = quests_completed(log)
    lat = run.latency_s or [0.0]
    return {"seed": seed, "end_reason": run.end_reason, "quests_done": len(done), "quests": done,
            "actions": run.actions, "llm_calls": run.api_calls, "invalid_outputs": run.invalid_outputs,
            "combats_won": sum(1 for x in run.combats if x.get("cleared")), "combats": len(run.combats),
            "latency_mean_s": round(sum(lat) / len(lat), 2), "latency_max_s": round(max(lat), 2),
            "input_tokens_per_call": round(run.usage.get("input_tokens", 0) / max(1, run.api_calls)),
            "seconds": run.seconds}


def summarize(rows: Sequence[Mapping[str, Any]], quests_total: int, **meta: Any) -> Dict[str, Any]:
    n = max(1, len(rows))
    calls = max(1, sum(r["llm_calls"] for r in rows))
    return dict(meta, runs=len(rows), quests_total=quests_total,
                full_clear_rate=round(sum(r["quests_done"] >= quests_total > 0 for r in rows) / n, 3),
                quests_done_mean=round(sum(r["quests_done"] for r in rows) / n, 2),
                actions_mean=round(sum(r["actions"] for r in rows) / n, 1),
                invalid_rate=round(sum(r["invalid_outputs"] for r in rows) / calls, 3),
                latency_mean_s=round(sum(r["latency_mean_s"] for r in rows) / n, 2),
                end_reasons={k: sum(r["end_reason"] == k for r in rows) for k in sorted({r["end_reason"] for r in rows})},
                rows=list(rows))


def run_batch(client: PlayClient, backend: LlmBackend, seeds: Sequence[int], config: Optional[PilotConfig] = None,
              spec: Union[None, PlaySpec, Mapping[str, Any], str] = None, responder: Any = None, cls: Optional[str] = None,
              out_dir: Union[None, str, Path] = None, record: bool = False,
              on_run: Optional[Callable[[Dict[str, Any]], None]] = None, **meta: Any) -> Dict[str, Any]:
    cfg = config or PilotConfig()
    out = Path(out_dir) if out_dir else None
    if out:
        out.mkdir(parents=True, exist_ok=True)
    steps = str(out / "steps.jsonl") if (out and record) else None
    desc = client.description
    total = len(desc.get("quests") or [])
    rows = []
    for seed in seeds:
        rec = StepRecorder(steps, run="batch-%s-s%d" % (time.strftime("%Y%m%d-%H%M%S"), seed), source="model",
                           model=backend.label) if steps else None
        run_cfg = PilotConfig(**dict(cfg.__dict__, stateless=cfg.stateless or rec is not None))
        obs = client.reset(seed=seed, cls=cls)
        run = LlmPilot(client, responder=responder, config=run_cfg, backend=backend, spec=spec, recorder=rec).run(obs)
        log = client.get_log()
        row = row_of(seed, run, log)
        rows.append(row)
        if out:
            (out / ("run-s%d.json" % seed)).write_text(json.dumps(run.to_dict(), ensure_ascii=False, indent=2, default=str),
                                                    encoding="utf-8")
        if on_run:
            on_run(dict(row, quests_total=total))
    summary = summarize(rows, total, game="%s %s" % (desc.get("game"), desc.get("game_version")), llm=backend.label,
                        mode="stateless" if (cfg.stateless or record) else "conversation", cls=cls, record=steps, **meta)
    if out:
        (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    return summary


__all__ = ["run_batch", "summarize", "row_of", "parse_seeds", "quests_completed"]
