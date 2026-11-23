"""BT-QA Game SDK 명령.

    python -m btqa_game check --game 127.0.0.1:7777     # 게임 서버 프로토콜 적합성 검사
    python -m btqa_game example --port 7777              # 예제 게임(최소 아레나) 서버 실행
    python -m btqa_game version
"""
from __future__ import annotations

import argparse
import sys


def main(argv=None) -> int:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
        except Exception:
            pass
    p = argparse.ArgumentParser(prog="btqa_game", description="BT-QA Game SDK")
    sub = p.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("check", help="게임 서버 프로토콜 적합성 검사")
    c.add_argument("--game", required=True, help="HOST:PORT")
    c.add_argument("--timeout", type=float, default=60.0)
    e = sub.add_parser("example", help="예제 게임 서버 (btqa_game.examples.minimal_arena)")
    e.add_argument("--host", default="127.0.0.1")
    e.add_argument("--port", type=int, default=7777)
    sub.add_parser("version")
    args = p.parse_args(argv)

    if args.cmd == "check":
        from btqa_game.conformance import check

        host, port = args.game.rsplit(":", 1)
        return 0 if check(host, int(port), args.timeout) else 1
    if args.cmd == "example":
        from btqa_game import QaGameServer
        from btqa_game.examples.minimal_arena import MinimalArena

        server = QaGameServer(MinimalArena, args.host, args.port)
        print("LISTENING %d" % server.address[1], flush=True)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass
        return 0
    from btqa_game import PROTOCOL_VERSION, __version__

    print("btqa_game %s (protocol %s)" % (__version__, PROTOCOL_VERSION))
    return 0


if __name__ == "__main__":
    sys.exit(main())
