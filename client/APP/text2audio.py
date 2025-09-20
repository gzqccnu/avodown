# avodown
# Copyright (c) 2025 gzqccnu 
# 
# This program is released under the terms of the Apache License.
# See https://opensource.org/licenses/Apache for more information.
# 
# Project homepage: https://github.com/gzqccnu/avodown
# Description: Using models to avoid the old fall down

import json
import threading
import time
from websocket import create_connection, WebSocketConnectionClosedException

# modify it to your server_ip
WS_URL = "ws://server_ip/ws/chat"

def recv_loop(ws):
    try:
        while True:
            raw = ws.recv()
            data = json.loads(raw)
            t = data.get("type")
            msg = data.get("message")
            print(f"🟢 recieve(type={t}): {msg}")
    except WebSocketConnectionClosedException:
        print("⚪ connection closed")
    except Exception as e:
        print("❌ recieve error:", e)

def main():
    print(f"▶️ connect to {WS_URL}")
    ws = create_connection(WS_URL)
    print("✅ connect successfully")

    register_msg = {
        "type": "register",
        "role": "ui"
    }
    ws.send(json.dumps(register_msg))
    print("📨 registerd as UI client")

    # 启动后台接收线程
    threading.Thread(target=recv_loop, args=(ws,), daemon=True).start()

    try:
        while True:
            user_input = input("pleae input text (enter exit to exit)：").strip()
            if not user_input:
                continue
            if user_input.lower() in ("exit", "quit"):
                break

            payload = {
                "type":    "text",
                "message": user_input
            }
            ws.send(json.dumps(payload, ensure_ascii=False))
            print("▶️ send：", json.dumps(payload, ensure_ascii=False))
    except KeyboardInterrupt:
        pass
    finally:
        print("🔒 close connection")
        ws.close()

if __name__ == "__main__":
    main()
