import os, time, base64, cv2, numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from modelLoader import FallDetectionModel
import asyncio

app = FastAPI()
model = FallDetectionModel()
# 启动时加载模型
@app.on_event("startup")
def load_models():
    model.load(
        weights=os.getenv("YOLO_WEIGHTS", "models/yolo.pt"),
        device=os.getenv("DEVICE", "cpu"),
        use_npu=os.getenv("USE_NPU", "0") == "1",
        img_size=int(os.getenv("IMG_SIZE", "640"))
    )

# Connection Manager 用于广播给所有前端订阅者
class ConnectionManager:
    def __init__(self):
        self.active: list[WebSocket] = []

    async def connect(self, ws: WebSocket):
        await ws.accept()
        self.active.append(ws)

    def disconnect(self, ws: WebSocket):
        self.active.remove(ws)

    async def broadcast(self, msg: dict):
        living = []
        for ws in self.active:
            try:
                await ws.send_json(msg)
                living.append(ws)
            except:
                pass  # 断开的客户端自动剔除
        self.active = living

manager = ConnectionManager()

# -------------------
# 树莓派上传端点
# -------------------
@app.websocket("/ws/upload")
async def ws_upload(ws: WebSocket):
    await ws.accept()
    try:
        while True:
            data = await ws.receive_text()
            # 解码 Base64 → BGR ndarray
            jpg = base64.b64decode(data)
            arr = np.frombuffer(jpg, dtype=np.uint8)
            frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if frame is None:
                continue

            # 推理
            result = model.predict_with_fall(
                frame,
                conf_thres=0.5,
                iou_thres=0.45,
                augment=False,
                classes=[0],
                agnostic_nms=False
            )
            result["timestamp"] = time.time()

            # 在 frame 上画框
            for d in result["detections"]:
                x1,y1,x2,y2 = d["box"]
                color = (0,0,255) if result["fall_detected"] else (0,255,0)
                cv2.rectangle(frame, (x1,y1),(x2,y2), color, 2)
                cv2.putText(
                    frame,
                    f"{d['label']}:{d['confidence']:.2f}",
                    (x1, y1 - 8),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    color,
                    2
                )

            # 编 JPEG→Base64
            _, buf = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY),70])
            b64f = base64.b64encode(buf).decode('ascii')
            msg = {
                "frame": b64f,
                "detections": result["detections"],
                "fall_detected": result["fall_detected"],
                "timestamp": result["timestamp"]
            }

            # 发回给树莓派（可选）
            await ws.send_json(msg)
            # 同时广播给所有前端
            await manager.broadcast(msg)

    except WebSocketDisconnect:
        print("Upload client disconnected")

# -------------------
# 浏览器订阅端点
# -------------------
@app.websocket("/ws/stream")
async def ws_stream(ws: WebSocket):
    await manager.connect(ws)
    try:
        while True:
            # 订阅端通常不发数据，只保持连接
            await ws.receive_text()
    except WebSocketDisconnect:
        manager.disconnect(ws)
        print("Stream client disconnected")

@app.websocket("/ws/detection")
async def ws_stream(ws: WebSocket):
    # 1. 接受并加入广播列表
    await manager.connect(ws)
    try:
        # 2. 只要连接活着，就保持挂起状态
        while True:
            # 这里不等待前端任何消息，仅周期性休眠，随时能响应 manager.broadcast
            await asyncio.sleep(1)
    except WebSocketDisconnect:
        # 3. 客户端断开时，从列表移除
        manager.disconnect(ws)
        print("Stream client disconnected")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("webVideo:app", host="127.0.0.1", port=7890)
