"""RFT — 게임을 채점기로 쓰는 강화 미세조정 (GRPO 계열, LoRA). 교사 없이 학생이 스스로 개선.

    train_rft(base="distill_out/r8/merged", game="127.0.0.1:7801", scratch="127.0.0.1:7802",
              episodes=[Episode(seed=11), Episode(seed=12, cls="c2")], reward_fn=my_reward, prepare=my_loadout,
              out_dir="distill_out/r10", sft="distill_out/data/v12/train.jsonl")

한 단계:
  1. 지금 상태(= 판 시작 조건 + 지금까지 고른 행동열 prefix)의 단계 입력 → 학생이 행동 K개 샘플 (+ 탐욕 1개는 기록·진행용)
  2. 분기 평가: 시험용 게임에서 같은 상태를 만들고(snapshot 또는 seed+prefix 재생, btqa_play.branch) 후보를 실행 → reward_fn
  3. 갱신 (샘플 K개만, 탐욕 출력은 그룹에 넣지 않는다 — 넣으면 기준이 한쪽으로 기운다):
       A = r − mean(r)                         (adv_norm="none", 기본)   또는 (r − mean)/(std+ε) ("std")
       손실 = −A·log π(후보)/|후보| + kl·k3(π‖π_ref) + sft_coef·NLL(SFT 샘플)
     - std 로 나누지 않는 이유: 점수가 0 과 0.2 로만 갈려도 ±1 로 커져 작은 차이를 크게 배운다 (AI_QA r10·r11)
     - k3 = exp(lr−lp) − (lr−lp) − 1 ≥ 0 (π_ref = 어댑터 끈 시작 모델)
     - SFT 샘플 NLL 을 섞어 SFT 에서 멀어지지 않게 (AI_QA r11: 한 판 39단계·8회 갱신만으로 무너짐)
     - 최고 − 최저 < min_gap 이면 갱신 안 함 (잡음)
  4. 진행: follow="best"(점수 최고 후보) / "greedy"(탐욕 출력) / "sample"(첫 샘플)로 본 게임을 한 걸음
  5. 판이 끝날 때마다 체크포인트 (ckpt-<n>/) — **채택은 평가 스위트로** (btqa_play.suite.compare), 훈련 보상이 아니다
reward_fn 은 **게임 쪽이 준다** (무엇이 진척인지는 게임 판단 — SDK 에 게임 내용 없음).

필요: torch·transformers·peft (venv_distill), 게임 인스턴스 2개 (main·scratch, 같은 빌드).
"""
from __future__ import annotations

import json
import random
import statistics
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple, Union

from btqa_play.actions import ActionHistory, action_system_prompt, last_action_record, parse_action, with_goal
from btqa_play.branch import Brancher, Episode, PrepareFn, call_prepare
from btqa_play.prompt import build_system_prompt, default_goal
from btqa_play.spec import PlaySpec, as_spec, effective_observation_drop
from btqa_play.tools import GameTools, build_tools, compact_observation

#: reward_fn(obs_before, tool, input, out) -> float.  out = GameTools.execute 결과 (status·reason·observation·combats) 또는 {"error"}
RewardFn = Callable[[Mapping[str, Any], str, Mapping[str, Any], Mapping[str, Any]], float]


def as_episodes(seeds: Sequence[int] = (), cls: Optional[str] = None,
                episodes: Optional[Sequence[Union[Episode, Mapping[str, Any]]]] = None) -> List[Episode]:
    if episodes:
        return [e if isinstance(e, Episode) else Episode(seed=int(e.get("seed", 1)), cls=e.get("class") or e.get("cls") or cls,
                                                        options=dict(e.get("options") or {}), prepare=dict(e.get("prepare") or {}))
                for e in episodes]
    return [Episode(seed=int(s), cls=cls) for s in seeds]


def train_rft(base: str, game: str, scratch: str, reward_fn: RewardFn, out_dir: Union[str, Path],
              seeds: Sequence[int] = (), episodes: Optional[Sequence[Union[Episode, Mapping[str, Any]]]] = None,
              prepare: Optional[PrepareFn] = None, spec: Union[None, PlaySpec, Mapping[str, Any], str] = None,
              cls: Optional[str] = None, k: int = 4, max_steps: int = 45, temperature: float = 0.9, lr: float = 5e-6,
              kl_coef: float = 0.05, adv_norm: str = "none", min_gap: float = 0.1, follow: str = "best",
              sft: Optional[Union[str, Path]] = None, sft_coef: float = 0.1, passes: int = 1,
              lora_r: int = 16, lora_alpha: int = 32, max_new_tokens: int = 160,
              dtype_name: str = "bfloat16", seed: int = 0, history: int = 0) -> Path:
    import torch
    from btqa_distill.policy import LoraPolicy, k3_kl, sft_nll
    from btqa_play.dataset import read_samples

    if adv_norm not in ("none", "std"):
        raise ValueError("adv_norm 은 none|std")
    if follow not in ("best", "greedy", "sample"):
        raise ValueError("follow 는 best|greedy|sample")
    if k < 2:
        raise ValueError("k ≥ 2 (그룹 안 비교)")
    eps = as_episodes(seeds, cls, episodes)
    if not eps:
        raise ValueError("학습 판(seeds 또는 episodes)이 없다")
    random.seed(seed)
    torch.manual_seed(seed)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    pol = LoraPolicy.load(base, lora_r=lora_r, lora_alpha=lora_alpha, dtype_name=dtype_name, train=True)
    params = pol.params
    opt = torch.optim.AdamW(params, lr=lr)
    sft_rows = read_samples(sft) if sft else []
    sp = as_spec(spec)

    main = Brancher(game, prepare, use_snapshot=False)   # 본 판은 그냥 진행
    br = Brancher(scratch, prepare)
    desc = main.desc
    tl = build_tools(desc, sp)
    names = [t["name"] for t in tl]
    system = action_system_prompt(with_goal(build_system_prompt(desc, sp), default_goal(desc, sp)), tl)
    drop = effective_observation_drop(desc, sp)

    log: Dict[str, Any] = {"base": base, "episodes_cfg": [e.__dict__ for e in eps], "k": k, "lr": lr, "kl": kl_coef,
                           "adv_norm": adv_norm, "min_gap": min_gap, "follow": follow, "sft": str(sft) if sft else None,
                           "sft_coef": sft_coef, "passes": passes, "history": history, "episodes": []}
    t0, n_ep = time.time(), 0
    for p in range(passes):
        for ep in eps:
            tools = main._start(ep)
            obs = dict(main.state_obs)
            prefix: List[Tuple[str, Dict[str, Any]]] = []
            hist = ActionHistory(history)
            rec_ep: Dict[str, Any] = {"pass": p, "episode": ep.__dict__, "steps": [], "updates": 0, "greedy_return": 0.0}
            for t in range(max_steps):
                user = hist.user(compact_observation(obs, drop))
                ids = pol.prompt_ids([{"role": "system", "content": system}, {"role": "user", "content": user}])
                greedy = pol.sample(ids, greedy=True, max_new_tokens=max_new_tokens)[0]
                samples = pol.sample(ids, n=k, temperature=temperature, max_new_tokens=max_new_tokens)
                texts = [greedy] + samples
                acts = [parse_action(x, names)[0] for x in texts]
                valid = [(i, a) for i, a in enumerate(acts) if a is not None and a["tool"] != "finish"]
                outs = dict(zip([i for i, _ in valid], br.try_actions(ep, prefix, [(a["tool"], a["input"]) for _, a in valid])))
                rewards = []
                for i, a in enumerate(acts):
                    if a is None:
                        rewards.append(-1.0)            # 형식 오류
                    elif a["tool"] == "finish":
                        rewards.append(float(reward_fn(obs, "finish", a["input"], {"status": "finish"})))
                    else:
                        rewards.append(float(reward_fn(obs, a["tool"], a["input"], outs[i])))
                g_rewards = rewards[1:]                 # 갱신 그룹 = 샘플만
                gap = max(g_rewards) - min(g_rewards)
                updated = False
                if gap >= min_gap and gap > 1e-6:
                    mu = statistics.mean(g_rewards)
                    sd = statistics.pstdev(g_rewards) or 0.0
                    opt.zero_grad()
                    for txt, r in zip(samples, g_rewards):
                        a_ids = pol.answer_ids(txt)
                        lp = pol.logp(ids, a_ids)
                        with pol.reference():
                            lr_ = pol.logp(ids, a_ids)
                        adv = (r - mu) / (sd + 1e-6) if adv_norm == "std" else (r - mu)
                        loss = (-adv * lp + kl_coef * k3_kl(lp, lr_, torch)) / len(samples)
                        loss.backward()
                    if sft_rows and sft_coef > 0:
                        (sft_coef * sft_nll(pol, random.choice(sft_rows))).backward()
                    torch.nn.utils.clip_grad_norm_(params, 1.0)
                    opt.step()
                    updated = True
                    rec_ep["updates"] += 1
                if follow == "greedy":
                    pick = 0
                elif follow == "sample":
                    pick = 1
                else:
                    pick = max(range(len(texts)), key=lambda i: rewards[i])
                act = acts[pick]
                rec_ep["greedy_return"] += rewards[0]
                rec_ep["steps"].append({"t": t, "greedy": rewards[0], "samples": [round(x, 2) for x in g_rewards],
                                        "updated": updated, "chosen": act and act["tool"]})
                print(json.dumps({"ep": n_ep, "seed": ep.seed, "t": t, "greedy": rewards[0],
                                  "samples": [round(x, 2) for x in g_rewards], "upd": updated,
                                  "chosen": act and act["tool"], "sec": round(time.time() - t0)}, ensure_ascii=False), flush=True)
                if act is None or act["tool"] == "finish":
                    break
                res = tools.execute(act["tool"], dict(act["input"]))
                prefix.append((act["tool"], dict(act["input"])))
                hist.push(last_action_record(act["tool"], act["input"], res))
                obs = tools.last_observation or obs
            rec_ep["greedy_return"] = round(rec_ep["greedy_return"], 3)
            log["episodes"].append(rec_ep)
            pol.save(str(out / ("ckpt-%d" % n_ep)))      # 판마다 체크포인트 — 채택은 평가 스위트로
            n_ep += 1
            (out / "rft_log.json").write_text(json.dumps(log, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    pol.save(str(out / "adapter"))
    log["seconds"] = round(time.time() - t0, 1)
    log["replays"] = br.replays
    log["snapshot"] = br.use_snapshot
    (out / "rft_log.json").write_text(json.dumps(log, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    main.close()
    br.close()
    return out / "adapter"


__all__ = ["train_rft", "as_episodes", "Brancher", "Episode", "RewardFn", "PrepareFn", "call_prepare"]
