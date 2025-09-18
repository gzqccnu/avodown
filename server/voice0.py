import time
import sounddevice as sd
import vosk
import queue
import json
from datetime import datetime
import asyncio
import io
from pydub import AudioSegment
from pydub.playback import play
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch


# 语音合成部分（保持不变）
async def text_to_speech(text, voice='zh-CN-XiaoxiaoNeural'):
    """将文本转为语音并直接播放"""
    import edge_tts
    communicate = edge_tts.Communicate(text=text, voice=voice)
    audio_data = bytearray()
    async for chunk in communicate.stream():
        if chunk["type"] == "audio":
            audio_data.extend(chunk["data"])
    audio = AudioSegment.from_mp3(io.BytesIO(audio_data))
    play(audio)


def speak(text, voice='zh-CN-XiaoxiaoNeural'):
    """同步调用语音合成接口"""
    asyncio.run(text_to_speech(text, voice))


class QwenClient:
    def __init__(self, model_path):
        # 加载本地微调好的Qwen模型
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"正在加载Qwen模型，使用设备: {self.device}")

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            device_map="auto",
            trust_remote_code=True
        ).eval()

        # 对话历史
        self.conversation_history = []
        # 系统提示词
        self.system_prompt = "你是一个智能语音助手，回答要简洁自然"

        print("Qwen模型加载完成")

    def chat(self, user_input):
        """与Qwen大模型对话"""
        self.conversation_history.append({
            "role": "user",
            "content": user_input
        })

        # 构造完整的对话历史（包含系统提示）
        messages = [
                       {"role": "system", "content": self.system_prompt}
                   ] + self.conversation_history[-6:]  # 保留最近3轮对话

        try:
            response, _ = self.model.chat(
                self.tokenizer,
                messages,
                history=None,
                temperature=0.7  # 控制生成随机性
            )

            self.conversation_history.append({
                "role": "assistant",
                "content": response
            })
            return response
        except Exception as e:
            print(f"大模型调用失败: {e}")
            return "抱歉，我暂时无法处理这个问题"


# 语音识别配置（保持不变）
MODEL_PATH = "./qwen_1.5b_chat_model"
SAMPLE_RATE = 16000
WAKE_WORD = "小助手"
CHUNK_SIZE = 4096

print("初始化语音识别模型...")
model = vosk.Model(MODEL_PATH)
recognizer = vosk.KaldiRecognizer(model, SAMPLE_RATE)
recognizer.SetWords(True)

audio_queue = queue.Queue()


def audio_callback(indata, frames, time, status):
    """音频采集回调函数"""
    if status:
        print(f"音频流状态: {status}")
    audio_queue.put(bytes(indata))


def wait_for_wake_word():
    """等待唤醒词"""
    print(f"\n等待唤醒词: '{WAKE_WORD}'... (按 Ctrl+C 停止)")
    audio_queue.queue.clear()

    while True:
        data = audio_queue.get()
        if recognizer.AcceptWaveform(data):
            result = json.loads(recognizer.Result())
            text = result.get("text", "").strip()
            if text and WAKE_WORD in text:
                print(f"\n唤醒成功! 检测到唤醒词: {text}")
                return True
        else:
            partial = json.loads(recognizer.PartialResult())
            partial_text = partial.get("partial", "").strip()
            if partial_text:
                print(f"正在监听: {partial_text}", end='\r')


def listen_for_speech(timeout_sec=5):
    """监听用户语音输入"""
    print("\n请说话...")
    start_time = datetime.now()
    audio_data = bytearray()

    while True:
        elapsed = (datetime.now() - start_time).total_seconds()
        if elapsed > timeout_sec:
            print("\n检测超时，未收到语音输入")
            return None

        try:
            data = audio_queue.get(timeout=2)
            audio_data.extend(data)
            if recognizer.AcceptWaveform(data):
                result = json.loads(recognizer.Result())
                text = result.get("text", "").strip()
                if text:
                    print(f"识别到语音: {text}")
                    return text
            else:
                partial = json.loads(recognizer.PartialResult())
                partial_text = partial.get("partial", "").strip()
                if partial_text:
                    print(f"正在识别: {partial_text}", end='\r')
        except queue.Empty:
            continue


def main():
    """主程序"""
    # 初始化Qwen客户端，指定本地模型路径
    qwen_model_path = "/mnt/workspace/.cache/modelscope/qwen_lora"  # 替换为你的模型路径
    ai_client = QwenClient(qwen_model_path)

    try:
        with sd.RawInputStream(
                samplerate=SAMPLE_RATE,
                blocksize=CHUNK_SIZE,
                dtype='int16',
                channels=1,
                callback=audio_callback
        ) as stream:

            while True:
                # 等待唤醒词
                wait_for_wake_word()

                # 播放响应时暂停录音
                stream.stop()
                speak("我在！")
                time.sleep(1)  # 等待语音播放完成
                stream.start()

                # 进入对话循环
                print("\n进入对话模式... (说'退出'结束对话)")
                while True:
                    user_input = listen_for_speech()

                    if user_input is None:
                        print("返回唤醒模式")
                        continue

                    if "退出" in user_input or "结束" in user_input:
                        stream.stop()
                        speak("好的，再见！")
                        time.sleep(1)
                        stream.start()
                        print("结束对话")
                        break

                    # 获取大模型回复
                    response = ai_client.chat(user_input)
                    print(f"助手回复: {response}")

                    # 播放回复时暂停录音
                    stream.stop()
                    speak(response)
                    time.sleep(0.5)
                    stream.start()

                    # 清空音频队列
                    audio_queue.queue.clear()

    except KeyboardInterrupt:
        print("\n程序终止")
    except Exception as e:
        print(f"程序异常: {e}")


if __name__ == "__main__":
    print("启动智能语音助手系统...")
    main()
