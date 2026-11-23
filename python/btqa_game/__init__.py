"""BT-QA SDK — 게임 쪽 (Python). 파이썬 게임이 학습 플랫폼에 붙기 위한 키트.

    from btqa_game import EpisodeHost, GameDecl, ClassDecl, ParamDecl, BlockDecl, TriggerDecl, QaGameServer

규칙: 학습 대상 = 블록 가중치뿐 (`declaration.py`). 계약: docs/PROTOCOL.md. Unity 판: unity/com.btqa.game. 적합성 검사: python -m btqa_game check --game HOST:PORT.
학습 쪽(balance_qa)·torch 를 import 하지 않는다.
"""
from btqa_game.declaration import (
    GROUP_BT,
    GROUP_PERSONA,
    WEIGHT_UNIT,
    BlockDecl,
    ClassDecl,
    DeclarationError,
    GameDecl,
    InitialStateDecl,
    ParamDecl,
    ScenarioDecl,
    TriggerDecl,
)
from btqa_game.host import EpisodeHost, EpisodeRequest, TriggerChannel
from btqa_game.protocol import PROTOCOL_VERSION, ProtocolError
from btqa_game.server import QaGameServer
from btqa_game.session import GameSession

__version__ = "0.2.0"

__all__ = [
    "PROTOCOL_VERSION", "ProtocolError",
    "GROUP_BT", "GROUP_PERSONA", "WEIGHT_UNIT",
    "ParamDecl", "BlockDecl", "ClassDecl", "TriggerDecl", "ScenarioDecl", "InitialStateDecl", "GameDecl", "DeclarationError",
    "EpisodeHost", "EpisodeRequest", "TriggerChannel", "GameSession", "QaGameServer",
]
