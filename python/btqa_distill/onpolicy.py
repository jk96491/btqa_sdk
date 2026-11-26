"""on-policy 선호 쌍 — 학생이 실제로 간 상태에서 **학생 자신의 샘플 K개**를 게임으로 채점해 (좋은 것, 나쁜 것) 쌍을 만든다.

    python -m btqa_distill pairs-onpolicy --base distill_out/r8/merged --game 127.0.0.1:7802 --records reports/eval-*/steps.jsonl ^
        --reward mygame.distill:reward [--prepare mygame.distill:prepare] --k 6 --margin 0.5 --out pairs_onpolicy.jsonl

왜 (AI_QA 2026-11-25 r9): 교정 기록 쌍(chosen = 교사 글, rejected = 학생 글)은 18개뿐이었고 chosen 이 학생 분포 밖이라
DPO 가 행동보다 글 차이를 밀었다 → 1/5 로 후퇴. on-policy 쌍은
  - 두 글 모두 학생이 쓴 것 (형식·말투 같음 → 차이는 행동),
  - 학생이 실제로 가는 상태 (스위트 train 묶음 실행·DAgger 기록의 steps.jsonl — 평가 묶음 기록은 쓰지 않는다: 평가 상태 누수),
  - 교사 없이 게임 보상으로 채점 → 상태 수만큼 늘릴 수 있다.

절차 (상태 하나):
  1. 기록에서 그 단계의 prefix(그 전에 실행된 행동열)와 시작 조건(episode: seed·class·options)을 읽는다
  2. 시험용 게임에서 그 상태를 만든다 (snapshot 또는 seed+prefix 재생) — 관찰이 기록과 다르면 건너뛴다 (재생 불일치)
  3. 기록된 프롬프트로 학생이 K개 샘플 → 행동별로 실행해 reward 훅으로 채점 (같은 행동은 한 번)
  4. 최고 − 최저 ≥ margin 이면 (최고 글, 최저 글) 쌍. 형식이 깨진 샘플은 쓰지 않는다 (형식은 JSON 강제가 맡는다)
reward 훅은 게임 쪽 (btqa_distill.rft.RewardFn 과 같은 모양). 1단계 보상이라 "진척"을 보상이 제대로 정의해야 한다 (해킹 주의).
"""
from __future__ import annotations

import json
import random
import time
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Union

from btqa_play.actions import parse_action
from btqa_play.branch import Brancher, Episode, PrepareFn, prefixes
from btqa_play.dataset import PAIR_FORMAT
from btqa_play.record import episode_of, read_steps, with_history
from btqa_play.spec import PlaySpec, as_spec, effective_observation_drop
from btqa_play.tools import build_tools, compact_observation


def _runs(paths: Sequence[Union[str, Path]]) -> Dict[str, List[Dict[str, Any]]]:
    runs: Dict[str, List[Dict[str, Any]]] = {}
    for p in paths:
        for r in read_steps(p):
            runs.setdefault(str(r.get("run")), []).append(r)
    for v in runs.values():
        v.sort(key=lambda r: int(r.get("step") or 0))
    return runs


def _same_obs(user_text: str, obs: Mapping[str, Any], drop: Sequence[str]) -> bool:
    try:
        recorded = json.loads(user_text).get("observation")
    except ValueError:
        return False
    now = json.loads(json.dumps(compact_observation(obs, drop), ensure_ascii=False))
    ign = ("seq", "game_time", "events")
    return {k: v for k, v in (recorded or {}).items() if k not in ign} == {k: v for k, v in now.items() if k not in ign}


def build_onpolicy_pairs(base: str, game: str, records: Sequence[Union[str, Path]], reward_fn: Any,
                         out_path: Union[str, Path], prepare: Optional[PrepareFn] = None,
                         spec: Union[None, PlaySpec, Mapping[str, Any], str] = None, cls: Optional[str] = None,
                         prepare_args: Optional[Mapping[str, Any]] = None, k: int = 6, temperature: float = 0.9,
                         margin: float = 0.5, max_states: int = 300, every: int = 1, include_target: bool = False,
                         max_new_tokens: int = 160, dtype_name: str = "bfloat16", seed: int = 0,
                         history: Optional[int] = None) -> Dict[str, Any]:
    from btqa_distill.policy import LoraPolicy

    random.seed(seed)
    pol = LoraPolicy.load(base, dtype_name=dtype_name, train=False)
    pol.torch.manual_seed(seed)
    br = Brancher(game, prepare)
    sp = as_spec(spec)
    names = [t["name"] for t in build_tools(br.desc, sp)]
    drop = effective_observation_drop(br.desc, sp)
    runs = _runs(records)
    if history is not None:   # 학생 입력 형식 = 학습한 형식 (record.with_history)
        runs = {k: with_history(v, history) for k, v in runs.items()}
    states = []
    for run, recs in runs.items():
        for i, (r, pre) in enumerate(zip(recs, prefixes(recs))):
            if i % max(1, every) == 0 and len(r.get("messages") or []) == 2:
                states.append((run, r, pre))
    random.shuffle(states)
    states = states[:max_states]
    stats: Dict[str, Any] = {"states": len(states), "pairs": 0, "replay_mismatch": 0, "no_margin": 0, "too_few_valid": 0,
                             "no_seed": 0, "invalid_samples": 0, "samples": 0, "reward_hist": {}}
    t0 = time.time()
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for n, (run, r, pre) in enumerate(states):
            e = episode_of(r)
            if "seed" not in e:
                stats["no_seed"] += 1
                continue
            ep = Episode(seed=int(e["seed"]), cls=e.get("class") or cls, options=dict(e.get("options") or {}),
                         prepare=dict(prepare_args or {}))
            obs = br.observation_at(ep, pre)
            if not _same_obs(r["messages"][1]["content"], obs, drop):
                stats["replay_mismatch"] += 1
                continue
            ids = pol.prompt_ids(r["messages"])
            texts = pol.sample(ids, n=k, temperature=temperature, max_new_tokens=max_new_tokens)
            if include_target and r.get("target_text"):
                texts.append(r["target_text"])
            stats["samples"] += len(texts)
            cands = []
            for t in texts:
                act, err = parse_action(t, names)
                if act is None:
                    stats["invalid_samples"] += 1
                    continue
                cands.append((t, act))
            if len(cands) < 2:
                stats["too_few_valid"] += 1
                continue
            outs = br.try_actions(ep, pre, [(a["tool"], a["input"]) for _, a in cands])
            scored = []
            for (t, a), o in zip(cands, outs):
                rw = float(reward_fn(obs, a["tool"], a["input"], {"status": "finish"} if a["tool"] == "finish" else o))
                scored.append((rw, t, a))
                key = "%.2f" % rw
                stats["reward_hist"][key] = stats["reward_hist"].get(key, 0) + 1
            hi = max(scored, key=lambda x: x[0])
            lo = min(scored, key=lambda x: x[0])
            if hi[0] - lo[0] < margin:
                stats["no_margin"] += 1
                continue
            pair = {"format": PAIR_FORMAT, "prompt": [dict(m) for m in r["messages"]], "chosen": hi[1], "rejected": lo[1],
                    "meta": {"run": run, "step": r.get("step"), "source": "onpolicy", "chosen_reward": hi[0],
                             "rejected_reward": lo[0], "chosen_tool": hi[2]["tool"], "rejected_tool": lo[2]["tool"],
                             "rewards": sorted(x[0] for x in scored)}}
            f.write(json.dumps(pair, ensure_ascii=False) + "\n")
            stats["pairs"] += 1
            if (n + 1) % 10 == 0:
                print(json.dumps({"state": n + 1, "pairs": stats["pairs"], "sec": round(time.time() - t0)}), flush=True)
    br.close()
    stats.update(seconds=round(time.time() - t0, 1), replays=br.replays, snapshot=br.use_snapshot, base=base,
                 k=k, temperature=temperature, margin=margin)
    out.with_suffix(".stats.json").write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
    return stats


__all__ = ["build_onpolicy_pairs"]
