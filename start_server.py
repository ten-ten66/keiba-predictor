#!/usr/bin/env python3
"""
競馬予想AI - 統合起動スクリプト
================================
FastAPIサーバー + リアルタイムジョブ を1プロセスで起動。

使い方:
  python start_server.py
  python start_server.py --port 8000
  
  # バックグラウンド
  nohup python start_server.py > /dev/null 2>&1 &
  
  # screen
  screen -S keiba-api
  python start_server.py
"""

import os
import sys
import argparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

DATA_DIR = os.environ.get("KEIBA_DATA_DIR", "data")
os.makedirs(DATA_DIR, exist_ok=True)


def main():
    parser = argparse.ArgumentParser(description="KEIBA AI サーバー起動")
    parser.add_argument("--port", type=int, default=8000, help="ポート番号")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="ホスト")
    parser.add_argument("--no-realtime", action="store_true", help="リアルタイムジョブを起動しない")
    args = parser.parse_args()

    print("🏇🏇🏇🏇🏇🏇🏇🏇🏇🏇🏇🏇🏇🏇🏇🏇")
    print("  KEIBA AI サーバー")
    print(f"  API: http://{args.host}:{args.port}")
    print(f"  Data: {DATA_DIR}")
    print("🏇🏇🏇🏇🏇🏇🏇🏇🏇🏇🏇🏇🏇🏇🏇🏇")

    # リアルタイムジョブ起動
    if not args.no_realtime:
        from realtime_job import run_as_thread
        run_as_thread()
        print("✅ リアルタイムジョブ起動")

    # APIサーバー起動
    import uvicorn
    uvicorn.run("api_server:app", host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
