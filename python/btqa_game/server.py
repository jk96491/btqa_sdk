"""TCP 게임 서버 — 게임 쪽 SDK. 인스턴스 1개 = 포트 1개 = 동시 연결 1개, 요청은 순차 처리.

    server = QaGameServer(lambda: MyHost(), port=7777)
    server.serve_forever()            # 또는 start_in_thread()
"""
from __future__ import annotations

import socket
import socketserver
import threading
from typing import Any, Callable, Dict, Optional, Tuple

from btqa_game import protocol as P
from btqa_game.host import EpisodeHost
from btqa_game.session import GameSession

HostFactory = Callable[[], EpisodeHost]


class _Handler(socketserver.StreamRequestHandler):
    server: "QaGameServer"

    def handle(self) -> None:
        self.request.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        session = GameSession(self.server.host_factory())

        def send(msg: Dict[str, Any]) -> None:
            self.wfile.write(P.encode(msg))
            self.wfile.flush()

        def recv() -> Optional[Dict[str, Any]]:
            while True:
                line = self.rfile.readline()
                if not line:
                    return None
                if line.strip():
                    return P.decode(line)

        while True:
            try:
                msg = recv()
            except P.ProtocolError as exc:
                send(P.error(exc.code, exc.message))
                continue
            except (ConnectionError, OSError):
                return
            if msg is None:
                return
            try:
                if not session.handle(msg, send, recv):
                    return
            except (ConnectionError, OSError):
                return


class QaGameServer(socketserver.TCPServer):
    allow_reuse_address = True

    def __init__(self, host_factory: HostFactory, host: str = "127.0.0.1", port: int = 7777) -> None:
        self.host_factory = host_factory
        host_factory().declaration().validate()   # 기동 시 정의서 규칙 검사 (가중치만 학습)
        super().__init__((host, port), _Handler)

    @property
    def address(self) -> Tuple[str, int]:
        h, p = self.server_address[:2]
        return str(h), int(p)

    def start_in_thread(self) -> threading.Thread:
        """테스트·노트북용: 백그라운드 스레드에서 serve_forever."""
        t = threading.Thread(target=self.serve_forever, name="btqa-game-%d" % self.address[1], daemon=True)
        t.start()
        return t

    def stop(self) -> None:
        self.shutdown()
        self.server_close()
