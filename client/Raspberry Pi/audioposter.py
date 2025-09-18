import asyncio, base64, json, queue
import sounddevice as sd
import websockets
from pydub import AudioSegment
from pydub.playback import play
from io import BytesIO

SERVER_WS   = "ws://111.173.104.220:9870/ws/chat"
SAMPLE_RATE = 16000
CHANNELS    = 1
DTYPE       = 'int16'
CHUNK       = 4096

audio_queue = queue.Queue()
is_playing  = False

def audio_callback(indata, frames, time, status):
    if status: print("音频状态：", status)
    if not is_playing:
        audio_queue.put(indata.copy())

async def audio_sender(ws):
    while True:
        pcm = await asyncio.to_thread(audio_queue.get)
        b64 = base64.b64encode(pcm.tobytes()).decode('ascii')
        await ws.send(json.dumps({"type":"audio","message":b64}))
        await asyncio.sleep(0)

async def message_handler(ws):
    global is_playing
    async for raw in ws:
        data = json.loads(raw)
        if data.get("type") == "audio":
            is_playing = True
            try:
                audio_bytes = base64.b64decode(data["message"])
                audio = AudioSegment.from_file(BytesIO(audio_bytes), format="mp3")
                play(audio)
            finally:
                with audio_queue.mutex: audio_queue.queue.clear()
                is_playing = False
        # 其余 type（text/asr/…）全忽略

async def run_client():
    async with websockets.connect(SERVER_WS) as ws:
        # 注册身份
        await ws.send(json.dumps({"type":"register","role":"pi"}))
        await ws.recv()  # （可选）等 ACK
        sender   = asyncio.create_task(audio_sender(ws))
        receiver = asyncio.create_task(message_handler(ws))
        await asyncio.wait([sender, receiver], return_when=asyncio.FIRST_EXCEPTION)

def main():
    stream = sd.InputStream(
        samplerate=SAMPLE_RATE, blocksize=CHUNK,
        dtype=DTYPE, channels=CHANNELS, callback=audio_callback
    )
    with stream:
        asyncio.run(run_client())

if __name__ == "__main__":
    main()
