# avodown
# Copyright (c) 2025 gzqccnu 
# 
# This program is released under the terms of the Apache License.
# See https://opensource.org/licenses/Apache for more information.
# 
# Project homepage: https://github.com/gzqccnu/avodown
# Description: Using models to avoid the old fall down

import cv2
import base64
import time
import threading
from websocket import create_connection, WebSocketConnectionClosedException
import json

# modify SERVER_URL to your server_ip
SERVER_URL = "ws://server_ip/ws/upload"
CAMERA_INDEX = 0  # camera index 0
JPEG_QUALITY = 80  # JPEG compress quality
FRAME_INTERVAL = 0.1  # second, sending frame interval


def send_frames():
    try:
        ws = create_connection(SERVER_URL)
        print(f"Connected to {SERVER_URL}")
    except Exception as e:
        print(f"Failed to connect to server: {e}")
        return

    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        print("Failed to open camera.")
        return

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Failed to read frame from camera.")
                break

            # JPEG encoding
            encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY]
            retval, buffer = cv2.imencode('.jpg', frame, encode_param)
            if not retval:
                print("Failed to encode frame.")
                continue

            # base64 serialization
            jpg_bytes = buffer.tobytes()
            b64_str = base64.b64encode(jpg_bytes).decode('utf-8')

            # send
            try:
                ws.send(b64_str)
            except WebSocketConnectionClosedException:
                print("WebSocket connection closed by server.")
                break

            # recieve result
            try:
                result = ws.recv()
                data = json.loads(result)
                print("Detections:", data["detections"], "Fall:", data["fall_detected"])
            except WebSocketConnectionClosedException:
                print("WebSocket connection closed while receiving.")
                break

            time.sleep(FRAME_INTERVAL)
    finally:
        cap.release()
        ws.close()
        print("Connection closed.")


if __name__ == '__main__':
    send_frames()
