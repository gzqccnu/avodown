# avodown
# Copyright (c) 2025 gzqccnu 
# 
# This program is released under the terms of the Apache License.
# See https://opensource.org/licenses/Apache for more information.
# 
# Project homepage: https://github.com/gzqccnu/avodown
# Description: Using models to avoid the old fall down

import os, json, traceback, base64, asyncio
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.websockets import WebSocketState
import vosk, soundfile as sf
from Qwen import QwenServer    # 你已有的支持 NPU 的 QwenServer
import edge_tts
import torch
import argparse

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--device",
        type=str,
        default=os.environ.get("DEVICE", "npu:1"),
        help="运行设备：cpu / cuda:0 / npu:1"
    )
    p.add_argument(
        "--vosk_model",
        type=str,
        default="../vosk_model",
        help="Vosk 模型路径"
    )
    p.add_argument(
        "--qwen_model",
        type=str,
        default="../qwen_1.5b_chat_model",
        help="Qwen2 模型路径"
    )
    return p.parse_args()


args = parse_args()

# 如果使用 NPU，需要先设置可见设备并初始化 compile 模式
if args.device.startswith("npu"):
    idx = args.device.split(":")[-1]
    os.environ["NPU_VISIBLE_DEVICES"] = idx
    torch.npu.set_compile_mode(jit_compile=True)
    torch.npu.config.allow_internal_format = False

# 初始化 VOSK 识别器（只支持 CPU）
print("加载 Vosk 模型...")
vosk_model = vosk.Model(args.vosk_model)
recognizer = vosk.KaldiRecognizer(vosk_model, 16000)
recognizer.SetWords(True)
print("Vosk 初始化完成")

# 初始化 Qwen 对话服务
print(f"加载 Qwen 模型到 {args.device} ...")
qwen = QwenServer(args.qwen_model, device=args.device)
print("QwenServer 加载完成")

# ——— TTS helper ———
async def tts_to_bytes(text:str, voice:str="zh-CN-XiaoxiaoNeural")->bytes:
    communicate = edge_tts.Communicate(text=text, voice=voice)
    audio = bytearray()
    async for chunk in communicate.stream():
        if chunk["type"]=="audio":
            audio.extend(chunk["data"])
    return bytes(audio)

# ——— FastAPI & CORS ———
app = FastAPI()
app.add_middleware(
  CORSMiddleware, allow_origins=["*"],
  allow_methods=["*"], allow_headers=["*"]
)


"""支持两种 JSON 消息：
1) {"type":"text",  "message":"你好"}
2) {"type":"audio","message":"<base64_pcm_chunk>"}
返回 JSON:
    {"type":"asr",   "message":"识别文本"}
    {"type":"text",  "message":"助手文本回复"}
    {"type":"audio","message":"<base64_tts_mp3>"}
"""
pi_clients = set()
ui_clients = set()

@app.websocket("/ws/chat")
async def websocket_chat(ws: WebSocket):
    await ws.accept()
    role = None
    qwen.history.clear()
    audio_buffer = b""
    try:
        # —— 注册阶段 —— 
        raw = await ws.receive_text()
        data = json.loads(raw)
        if data.get("type") == "register" and data.get("role") in ("pi", "ui"):
            role = data["role"]
            if role == "pi":
                pi_clients.add(ws)
            else:
                ui_clients.add(ws)
            await ws.send_json({"type":"register_ack","message":f"registered as {role}"})
        else:
            # 未注册就发来了别的消息，直接断
            await ws.close()
            return

        # —— 主循环 —— 
        buffer = b""
        while True:
            raw = await ws.receive_text()
            data = json.loads(raw)
            msg_type = data.get("type")
            msg_body = data.get("message","")

            # —— 来自 UI 的纯文本，合成 TTS，发给 Pi —— 
            if role == "ui" and msg_type == "text":
                text_to_play = msg_body.strip()
                audio_bytes = await tts_to_bytes(text_to_play)
                b64 = base64.b64encode(audio_bytes).decode("ascii")
                payload = {"type":"audio", "message":b64}
                # 只给 Pi 客户端推送
                for pi in list(pi_clients):
                    try:
                        await pi.send_json(payload)
                    except:
                        pi_clients.discard(pi)

            # —— 来自 Pi 的音频块，用 ASR+Qwen→回复，推给 pi —— 
            elif role == "pi" and msg_type == "audio":
                # 1) 解码 base64 得到 PCM bytes
                try:
                    pcm_chunk = base64.b64decode(msg_body)
                except Exception:
                    await ws.send_json({"type":"error","message":"invalid_audio_base64"})
                    continue

                # 2) 累积到 buffer，并用 Vosk 做 ASR
                audio_buffer += pcm_chunk
                if recognizer.AcceptWaveform(pcm_chunk):
                    result = json.loads(recognizer.Result())
                    asr_text = result.get("text", "").strip()
                    audio_buffer = b""  # 清空 buffer
                    if not asr_text:
                        continue

                    # 3) 用 Qwen 生成对话回复
                    reply_text = qwen.chat(asr_text)
                    qwen.history.clear()
                    # 4) TTS 合成回复文本
                    reply_audio_bytes = await tts_to_bytes(reply_text)

                    # 5) 推文字回复给 pi
                    txt = {"type":"text","message":reply_text}
                    for pi in list(pi_clients):
                        try:    await pi.send_json(txt)
                        except: pi_clients.discard(pi)

                    # 6) 再推音频回复给 pi
                    b64r = base64.b64encode(reply_audio_bytes).decode("ascii")
                    aud = {"type":"audio","message":b64r}
                    for pi in list(pi_clients):
                        try:    
                            await pi.send_json(aud)
                        except: pi_clients.discard(pi)
            
    except WebSocketDisconnect:
        pass
    except Exception:
        traceback.print_exc()
    finally:
        if role == "pi":
            pi_clients.discard(ws)
        elif role == "ui":
            ui_clients.discard(ws)
        
        # 关键修复：检查连接状态并捕获关闭异常
        if ws.application_state == WebSocketState.CONNECTED:
            try:
                await ws.close()
            except RuntimeError:
                # 忽略"已关闭"的异常
                pass

            
if __name__ == "__main__":
    import uvicorn

    # 将 device 信息打印到启动日志
    print(f"启动服务 (device={args.device}) ...")
    uvicorn.run(
        "webAudioQA:app",
        host="127.0.0.1",
        port=9870,
        reload=False,
        ws_ping_interval=10,
        ws_ping_timeout=30
    )
