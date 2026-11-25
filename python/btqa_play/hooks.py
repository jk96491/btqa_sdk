"""게임 쪽 함수 끼우기 — "모듈:이름" 문자열로 게임 고유 처리(전처리·보상·준비)를 SDK 명령에 넘긴다. 표준 라이브러리만.

SDK 는 게임 내용을 모른다. 증류에서 게임마다 다른 것 — 학습 목표 thought 전처리, RFT 보상, 판 시작 준비(로드아웃) — 은
게임 쪽 패키지가 함수로 두고, CLI 에는 `--prep mygame.distill:prep_sample` 처럼 이름만 준다. SDK 는 그 패키지를 import 하지 않는다
(실행할 때 이름으로 불러올 뿐).
"""
from __future__ import annotations

import importlib
from typing import Any, Callable, Optional


def load_hook(spec: Optional[str]) -> Optional[Callable[..., Any]]:
    """"pkg.mod:func" → 함수 (None 이면 None). 잘못된 이름은 ValueError."""
    if not spec:
        return None
    mod, sep, name = spec.partition(":")
    if not sep or not mod or not name:
        raise ValueError("훅 이름은 '모듈:함수' 형식: %r" % spec)
    fn = importlib.import_module(mod)
    for part in name.split("."):
        fn = getattr(fn, part)
    if not callable(fn):
        raise ValueError("훅이 함수가 아님: %r" % spec)
    return fn


__all__ = ["load_hook"]
