"""Entry point for the Finance RAG Strategy Comparison Platform.

Usage::

    python main.py

Then open http://localhost:8000 in your browser.
"""

from __future__ import annotations

import os
import socket
import uvicorn


def _kill_port(port: int) -> None:
    """Force-kill any process holding *port* (Windows only)."""
    if os.name != "nt":
        return
    import subprocess, sys
    try:
        # Find PID
        result = subprocess.run(
            ["netstat", "-ano"],
            capture_output=True, text=True,
        )
        pids = set()
        for line in result.stdout.splitlines():
            if f":{port}" in line and "LISTENING" in line:
                parts = line.strip().split()
                pids.add(parts[-1])
        # Kill each PID
        for pid in pids:
            subprocess.run(
                ["taskkill", "/F", "/PID", pid],
                capture_output=True,
            )
            print(f"[main] Killed PID {pid} on port {port}")
    except Exception as exc:
        print(f"[main] Failed to kill port {port}: {exc}", file=sys.stderr)


def _wait_port_free(port: int, timeout: float = 5.0) -> bool:
    """Return True once *port* is free, or False after *timeout* seconds."""
    import time
    deadline = time.time() + timeout
    while time.time() < deadline:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.settimeout(0.5)
        try:
            s.bind(("0.0.0.0", port))
            s.close()
            return True
        except OSError:
            time.sleep(0.3)
        finally:
            s.close()
    return False


def main():
    # Auto-clean stale port before starting
    _kill_port(8000)
    if _wait_port_free(8000):
        print("[main] Port 8000 is free, starting server...")
    else:
        print("[main] WARNING: Port 8000 still occupied, uvicorn may fail")

    # 挂载 Vue3 前端构建产物（若存在）
    _mount_frontend()

    uvicorn.run(
        "finance_rag.api:app",
        host="0.0.0.0",
        port=8000,
        reload=os.getenv("DEBUG", "false").lower() == "true",
        log_level="info",
    )


def _mount_frontend() -> None:
    """若 frontend/dist 存在，将其作为静态文件挂载到 FastAPI 应用。

    开发时前端由 Vite dev server（端口 5173）独立运行；
    生产部署时前端构建为 dist，由此函数挂载到后端同端口。
    """
    from pathlib import Path

    dist = Path(__file__).resolve().parent / "frontend" / "dist"
    if not dist.exists():
        print("[main] frontend/dist 不存在，前端请通过 Vite dev server 启动（npm run dev）")
        return

    # 延迟导入避免循环依赖
    from finance_rag.api import app
    from fastapi.staticfiles import StaticFiles

    app.mount("/", StaticFiles(directory=str(dist), html=True), name="frontend")
    print(f"[main] 已挂载前端静态文件：{dist}")


if __name__ == "__main__":
    main()
