"""启动 FastAPI 服务。

用法:
    python scripts/serve.py
    python scripts/serve.py --host 0.0.0.0 --port 8000
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import uvicorn


def main():
    ap = argparse.ArgumentParser(description="启动 PromptGuard FastAPI 服务")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--reload", action="store_true")
    args = ap.parse_args()

    print(f"PromptGuard Backend 服务启动: http://{args.host}:{args.port}")
    print("文档: http://%s:%d/docs" % (args.host, args.port))
    uvicorn.run("api.server:app", host=args.host, port=args.port,
                reload=args.reload)


if __name__ == "__main__":
    main()
