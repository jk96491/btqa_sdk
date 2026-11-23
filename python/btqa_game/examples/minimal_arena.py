"""최소 예제 게임 — SDK 사용법과 일반성 확인용 (QaArena 와 블록·파라미터 이름이 전혀 다르다).

1:1 결투, 1초 = 1스텝. BT 전략 선택 노드: S = N × w 가 가장 큰 블록 실행.
    attack  N = 0.6                                 가중치 w_attack
    guard   N = max(0, (caution − hp) / caution)    가중치 w_guard   (받는 피해 70% 감소)
    heal    N = 1 (hp < heal_hp, 물약 남음) / 제외   가중치 w_heal    (+40% 체력, 소모품 1)
학습 대상 = w_attack, w_guard, w_heal 뿐. heal_hp 는 기획 설정값, caution 은 페르소나 값.

    python -m btqa_game example --port 7777
    python -m btqa_game check --game 127.0.0.1:7777
"""
from __future__ import annotations

import random
from typing import Any, Dict, Optional

from btqa_game import (
    GROUP_PERSONA,
    BlockDecl,
    ClassDecl,
    EpisodeHost,
    EpisodeRequest,
    GameDecl,
    ParamDecl,
    ScenarioDecl,
    TriggerChannel,
    TriggerDecl,
)

TIME_LIMIT = 300.0
MAX_DEATHS = 2
STALL_SECONDS = 60
STATE = ("hp_ratio", "enemy_hp_ratio", "potions_ratio", "time_ratio")
TOTALS = ("elapsed", "deaths", "consumables_used", "boss_hp_remaining")
SCENARIOS = {"base": 1.0, "enemy_tough": 1.3}   # 적 체력 배율


class MinimalArena(EpisodeHost):
    def declaration(self) -> GameDecl:
        duelist = ClassDecl(
            "duelist",
            params=(
                ParamDecl("w_attack", 0.0, 1.0, 0.6, description="공격 블록 가중치"),
                ParamDecl("w_guard", 0.0, 1.0, 0.5, description="방어 블록 가중치"),
                ParamDecl("w_heal", 0.0, 1.0, 0.8, description="회복 블록 가중치"),
                ParamDecl("heal_hp", 0.1, 0.9, 0.4, unit="hp_ratio", description="물약을 마시는 체력"),
                ParamDecl("caution", 0.05, 1.0, 0.5, group=GROUP_PERSONA, unit="hp_ratio", description="방어 필요도가 오르기 시작하는 체력"),
            ),
            blocks=(BlockDecl("attack", "w_attack"), BlockDecl("guard", "w_guard"), BlockDecl("heal", "w_heal")),
            display_name="결투가",
        )
        return GameDecl(
            game="minimal_arena", game_version="0.1.0", time_limit=TIME_LIMIT, max_deaths=MAX_DEATHS,
            decision_interval=1.0, classes=(duelist,),
            scenarios=tuple(ScenarioDecl(n) for n in SCENARIOS),
            triggers=TriggerDecl("enemy_hp_below", (0.66, 0.33), STATE, TOTALS),
        )

    def run_episode(self, req: EpisodeRequest, triggers: TriggerChannel) -> Dict[str, Any]:
        rng = random.Random(req.seed)          # 판의 모든 난수는 이 seed 하나에서
        p = dict(req.params)
        enemy_max = 1000.0 * SCENARIOS[req.scenario]
        hp, enemy, potions, deaths, used = 1.0, enemy_max, 3, 0, 0
        t, last_hit, fired = 0.0, 0.0, 0
        pending = sorted(req.thresholds, reverse=True)

        def state() -> Dict[str, float]:
            return {"hp_ratio": hp, "enemy_hp_ratio": enemy / enemy_max, "potions_ratio": potions / 3.0,
                    "time_ratio": t / TIME_LIMIT}

        def totals() -> Dict[str, float]:
            return {"elapsed": t, "deaths": deaths, "consumables_used": used, "boss_hp_remaining": enemy / enemy_max}

        def fire() -> None:
            nonlocal fired
            p.update(triggers.fire(fired, state(), totals()))
            fired += 1

        if pending:
            fire()                               # index 0 = 판 시작 결정
        stalled = False
        while enemy > 0 and t < TIME_LIMIT and deaths <= MAX_DEATHS:
            scores = {"attack": 0.6 * p["w_attack"],
                      "guard": max(0.0, (p["caution"] - hp) / p["caution"]) * p["w_guard"]}
            if hp < p["heal_hp"] and potions > 0:
                scores["heal"] = 1.0 * p["w_heal"]
            best: Optional[str] = max((s, n) for n, s in scores.items() if s > 0)[1] if any(s > 0 for s in scores.values()) else None
            incoming = rng.uniform(0.02, 0.06)
            if best == "attack":
                enemy -= rng.uniform(15.0, 25.0)
                last_hit = t
            elif best == "guard":
                incoming *= 0.3
            elif best == "heal":
                hp, potions, used = min(1.0, hp + 0.4), potions - 1, used + 1
            hp -= incoming
            if hp <= 0:
                deaths, hp = deaths + 1, 0.5
            t += 1.0
            if t - last_hit >= STALL_SECONDS:
                stalled = True
                break
            crossed = [th for th in pending if enemy / enemy_max < th]
            if crossed and enemy > 0:
                pending = [th for th in pending if th not in crossed]
                fire()                           # 한 틱에 여러 경계를 넘어도 1번
        enemy = max(0.0, enemy)
        cleared = enemy <= 0 and deaths <= MAX_DEATHS and not stalled
        return {
            "seed": req.seed, "cleared": cleared, "elapsed": t, "clear_time": t if cleared else None,
            "boss_hp_remaining": enemy / enemy_max, "deaths": deaths, "consumables_used": used,
            "triggers_fired": fired, "stalled": stalled, "stall_reason": "60초 동안 공격 없음" if stalled else None,
            "invalid": False, "invalid_reason": None,
        }
