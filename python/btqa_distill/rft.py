"""RFT — 게임을 채점기로 쓰는 강화 미세조정 (GRPO 방식, LoRA). 교사 없이 학생이 스스로 개선.

    train_rft(base="distill_out/r4/merged", game="127.0.0.1:7801", scratch="127.0.0.1:7802", seeds=[11, 12],
              reward_fn=my_reward, prepare=my_loadout, out_dir="distill_out/r5")

한 단계:
  1. 지금 상태(= seed + 지금까지 고른 행동열 prefix)의 단계 입력 → 학생이 행동 K개 샘플 (+ greedy 1개)
  2. 분기 평가: scratch 게임을 reset(seed) 하고 prefix 를 재생하면 **같은 상태**가 된다 (게임 결정성, PLAY_PROTOCOL 4.4·4.8)
     → 후보 행동을 실행 → reward_fn(관찰 전, 후보, 결과) 로 점수
  3. GRPO: 그룹 안 상대 이점 A = (r − mean) / (std + ε), 손실 = −A·log π(후보) + kl·(log π − log π_ref)  (π_ref = 어댑터 끔)
  4. 가장 점수 높은 후보로 main 게임을 한 걸음 진행 (prefix 에 붙임) — 학생이 가 볼 상태를 넓혀 간다
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

from btqa_distill.config import QWEN_TARGETS, load_causal_lm
from btqa_distill.train import _ids
from btqa_play.actions import action_system_prompt, last_action_record, parse_action, step_input, with_goal
from btqa_play.client import PlayClient
from btqa_play.prompt import build_system_prompt, default_goal
from btqa_play.spec import PlaySpec, as_spec, effective_observation_drop
from btqa_play.tools import GameTools, build_tools, compact_observation

#: reward_fn(obs_before, tool, input, out) -> float.  out = GameTools.execute 결과 (status·reason·observation·combats) 또는 {"error"}
RewardFn = Callable[[Mapping[str, Any], str, Mapping[str, Any], Mapping[str, Any]], float]
#: prepare(client, desc) -> responder (로드아웃·전투 응답기)
PrepareFn = Callable[[PlayClient, Mapping[str, Any]], Any]


class Brancher:
    """scratch 게임에서 seed + prefix 재생으로 같은 상태를 만들고 후보 행동을 실행해 본다."""

    def __init__(self, endpoint: str, prepare: Optional[PrepareFn] = None, cls: Optional[str] = None) -> None:
        self.c = PlayClient.connect(endpoint)
        self.c.hello()
        self.desc = self.c.describe()
        self.responder = prepare(self.c, self.desc) if prepare else None
        self.cls = cls

    def try_action(self, seed: int, prefix: Sequence[Tuple[str, Dict[str, Any]]], tool: str,
                   inp: Mapping[str, Any]) -> Dict[str, Any]:
        self.c.reset(seed=seed, cls=self.cls)
        tools = GameTools(self.c, self.responder)
        for t, i in prefix:
            tools.execute(t, i)
        try:
            return tools.execute(tool, dict(inp))
        except Exception as e:   # 전투 중 오류 등
            return {"error": str(e)}

    def close(self) -> None:
        self.c.close()


def train_rft(base: str, game: str, scratch: str, seeds: Sequence[int], reward_fn: RewardFn, out_dir: Union[str, Path],
              prepare: Optional[PrepareFn] = None, spec: Union[None, PlaySpec, Mapping[str, Any], str] = None,
              cls: Optional[str] = None, k: int = 4, max_steps: int = 45, temperature: float = 0.9, lr: float = 1e-5,
              kl_coef: float = 0.05, lora_r: int = 16, lora_alpha: int = 32, max_new_tokens: int = 160,
              dtype_name: str = "bfloat16", seed: int = 0) -> Path:
    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import AutoTokenizer

    random.seed(seed)
    torch.manual_seed(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = getattr(torch, dtype_name)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    tok = AutoTokenizer.from_pretrained(base)
    model = load_causal_lm(base, dtype)
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    model = get_peft_model(model, LoraConfig(r=lora_r, lora_alpha=lora_alpha, lora_dropout=0.0,
                                             target_modules=list(QWEN_TARGETS), task_type="CAUSAL_LM"))
    for p in model.parameters():
        if p.requires_grad:
            p.data = p.data.float()
    model.to(device)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=lr)
    sp = as_spec(spec)

    main = Brancher(game, prepare, cls)
    br = Brancher(scratch, prepare, cls)
    desc = main.desc
    tl = build_tools(desc, sp)
    names = [t["name"] for t in tl]
    system = action_system_prompt(with_goal(build_system_prompt(desc, sp), default_goal(desc, sp)), tl)
    drop = effective_observation_drop(desc, sp)

    def prompt_ids(user: str) -> List[int]:
        return _ids(tok.apply_chat_template([{"role": "system", "content": system}, {"role": "user", "content": user}],
                                            tokenize=True, add_generation_prompt=True))

    def sample(ids: List[int], greedy: bool) -> str:
        x = torch.tensor([ids], device=device)
        model.eval()
        with torch.no_grad(), torch.autocast(device_type="cuda", dtype=dtype, enabled=device.startswith("cuda")):
            g = model.generate(x, attention_mask=torch.ones_like(x), max_new_tokens=max_new_tokens, do_sample=not greedy,
                               temperature=temperature if not greedy else None, top_p=0.95 if not greedy else None,
                               pad_token_id=tok.pad_token_id or tok.eos_token_id, use_cache=True)   # 체크포인팅이 캐시를 끄지 않게
        model.train()
        return tok.decode(g[0, len(ids):], skip_special_tokens=True)

    def logp(ids: List[int], answer_ids: List[int]) -> Any:
        x = torch.tensor([ids + answer_ids], device=device)
        with torch.autocast(device_type="cuda", dtype=dtype, enabled=device.startswith("cuda")):
            lg = model(input_ids=x, logits_to_keep=len(answer_ids) + 1).logits[:, :-1, :].float()   # 답 구간 logits 만
        return torch.log_softmax(lg, -1).gather(-1, x[:, len(ids):].unsqueeze(-1)).squeeze(-1).sum()

    log: Dict[str, Any] = {"base": base, "seeds": list(seeds), "k": k, "lr": lr, "kl": kl_coef, "episodes": []}
    t0 = time.time()
    for s in seeds:
        obs = main.c.reset(seed=s, cls=cls)
        tools = GameTools(main.c, main.responder)
        prefix: List[Tuple[str, Dict[str, Any]]] = []
        last: Optional[Dict[str, Any]] = None
        ep: Dict[str, Any] = {"seed": s, "steps": []}
        for t in range(max_steps):
            user = step_input(compact_observation(obs, drop), last)
            ids = prompt_ids(user)
            texts = [sample(ids, greedy=True)] + [sample(ids, greedy=False) for _ in range(k)]
            cands, rewards = [], []
            for txt in texts:
                act, err = parse_action(txt, names)
                if act is None:
                    cands.append((txt, None))
                    rewards.append(-1.0)            # 형식 오류
                    continue
                if act["tool"] == "finish":
                    r = reward_fn(obs, "finish", act["input"], {"status": "finish"})
                else:
                    r = reward_fn(obs, act["tool"], act["input"], br.try_action(s, prefix, act["tool"], act["input"]))
                cands.append((txt, act))
                rewards.append(float(r))
            mu, sd = statistics.mean(rewards), (statistics.pstdev(rewards) or 0.0)
            if sd > 1e-6:                                   # 모두 같은 점수면 배울 것이 없다
                opt.zero_grad()
                for (txt, _), r in zip(cands, rewards):
                    a_ids = _ids(tok(txt + (tok.eos_token or ""), add_special_tokens=False))
                    lp = logp(ids, a_ids)
                    with torch.no_grad(), model.disable_adapter():
                        lr_ = logp(ids, a_ids)
                    adv = (r - mu) / (sd + 1e-6)
                    loss = (-adv * lp + kl_coef * (lp - lr_)) / len(cands)
                    loss.backward()
                torch.nn.utils.clip_grad_norm_(params, 1.0)
                opt.step()
            best = max(range(len(cands)), key=lambda i: rewards[i])
            txt, act = cands[best]
            ep["steps"].append({"t": t, "rewards": [round(x, 2) for x in rewards], "chosen": act and act["tool"],
                                "greedy_reward": rewards[0]})
            print(json.dumps({"seed": s, "t": t, "rewards": ep["steps"][-1]["rewards"], "chosen": ep["steps"][-1]["chosen"],
                              "sec": round(time.time() - t0)}, ensure_ascii=False), flush=True)
            if act is None or act["tool"] == "finish":
                break
            res = tools.execute(act["tool"], dict(act["input"]))
            prefix.append((act["tool"], dict(act["input"])))
            last = last_action_record(act["tool"], act["input"], res)
            obs = tools.last_observation or obs
        log["episodes"].append(ep)
        model.save_pretrained(str(out / "adapter"))   # 판마다 저장 (중간에 멈춰도 남는다)
    tok.save_pretrained(str(out / "adapter"))
    log["seconds"] = round(time.time() - t0, 1)
    (out / "rft_log.json").write_text(json.dumps(log, ensure_ascii=False, indent=2), encoding="utf-8")
    main.close()
    br.close()
    return out / "adapter"


__all__ = ["train_rft", "Brancher", "RewardFn", "PrepareFn"]
