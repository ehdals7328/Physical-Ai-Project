"""Waitress WSGI 서버를 실행하는 애플리케이션 진입점입니다."""

from __future__ import annotations

import os

from waitress import serve

from app import create_app

app = create_app()


if __name__ == "__main__":
    host = os.getenv("HOST", "127.0.0.1")
    port = int(os.getenv("PORT", "5000"))
    threads = int(os.getenv("WAITRESS_THREADS", "8"))
    print("=" * 58)
    print(" Shipyard Physical AI Platform")
    print(f" URL: http://{host}:{port}")
    print(f" Waitress threads: {threads}")
    print("=" * 58)
    serve(app, host=host, port=port, threads=threads)
