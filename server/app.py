# avodown
# Copyright (c) 2025 gzqccnu <gzqccnu@gmail.com>
#
# This program is released under the terms of the Apache License.
# See https://opensource.org/licenses/Apache for more information.
#
# Project homepage: https://github.com/gzqccnu/avodown
# Description: Using models to avoid the old fall down

# app.py

import os
import time
from io import BytesIO

import av
import numpy as np
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from PIL import Image
import json
from fastapi.responses import StreamingResponse

from modelLoader import FallDetectionModel


# ----------------------
# Pydantic Schemas
# ----------------------
class Detection(BaseModel):
    label: str
    confidence: float
    box: list[int]

class FallResponse(BaseModel):
    detections: list[Detection]
    fall_detected: bool
    timestamp: float

class VideoResponse(BaseModel):
    frames: list[FallResponse]


# ----------------------
# App & Model Init
# ----------------------
app = FastAPI(title="Fall Detection API (PyAV)")

model = FallDetectionModel()

@app.on_event("startup")
def load_models():
    model.load(
        weights=os.getenv("YOLO_WEIGHTS", "models/yolo.pt"),
        device=os.getenv("DEVICE", "cpu"),
        use_npu=os.getenv("USE_NPU", "0") == "1",
        img_size=int(os.getenv("IMG_SIZE", "640"))
    )


# ----------------------
# 单图像推理接口
# ----------------------
@app.post("/predict", response_model=FallResponse)
async def predict_image(file: UploadFile = File(...)):
    data = await file.read()
    try:
        img = Image.open(BytesIO(data)).convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"无法解码图片: {e}")
    frame = np.array(img)[:, :, ::-1]  # RGB → BGR

    result = model.predict_with_fall(
        frame,
        conf_thres=0.5,
        iou_thres=0.45,
        augment=False,
        classes=[0],
        agnostic_nms=False
    )
    result["timestamp"] = time.time()
    # print("detection", result)
    return JSONResponse(content=result)


# ----------------------
# 视频推理接口（PyAV 全内存）
# ----------------------
@app.post("/predict_video", response_model=VideoResponse)
async def predict_video(file: UploadFile = File(...)):
    data = await file.read()
    try:
        container = av.open(BytesIO(data))
    except av.AVError as e:
        raise HTTPException(status_code=400, detail=f"无法解码视频流: {e}")

    frames_out = []
    for packet in container.demux(video=0):
        for frame in packet.decode():
            # 将 PyAV Frame 转为 BGR ndarray
            img = frame.to_ndarray(format="bgr24")
            res = model.predict_with_fall(
                img,
                conf_thres=0.5,
                iou_thres=0.45,
                augment=False,
                classes=[0],
                agnostic_nms=False
            )
            res["timestamp"] = time.time()
            print("detection", res)
            frames_out.append(res)
    return JSONResponse(content={"frames": frames_out})

@app.post("/predict_video_stream")
async def predict_video_stream(file: UploadFile = File(...)):
    data = await file.read()
    try:
        container = av.open(BytesIO(data))
    except av.AVError as e:
        raise HTTPException(400, f"无法解码视频流: {e}")

    def frame_generator():
        for packet in container.demux(video=0):
            for frame in packet.decode():
                img = frame.to_ndarray(format="bgr24")
                res = model.predict_with_fall(
                    img,
                    conf_thres=0.5,
                    iou_thres=0.45,
                    augment=False,
                    classes=[0],
                    agnostic_nms=False
                )
                res["timestamp"] = time.time()
                # 将字典编码成一行 JSON，然后加两回车符合 SSE 规范
                yield f"data: {json.dumps(res)}\n\n"
        # 流结束时发一个 special event
        yield "event: end\ndata: done\n\n"

    return StreamingResponse(
        frame_generator(),
        media_type="text/event-stream"
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="127.0.0.1", port=7890)
