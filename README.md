# avodown

Avoid the old fall down. <br>

`avo` is the abbreviation of "audio and video". So actually **avodown** is to avoid down.

> [!Note]
> This repository contains the source code of project **posecare**.


> [!Info]
> This project is mainly oriented to caring the old.

## Introduction

The project have three part. First is the server part mainly load the fall-detect-model and chat-model. Second is the edge part actually it is a small car made by ourselves on which have a voice broadcast module and a video camera. Finally is the app part. The server loads a **Qwen-1.5b model**, a **vosk model** and a **YOLO+OpenPose** model. The `edge` will follow the old automatically by radar, and will stream the video and audio to the `server` respectivelly. The `server` part recieves the video and uses `YOLO+OpenPose` to direct whether the old will fall or not. Then sending message to the `app`. The `app` installed on the young's phone will recieve that. If not fall, there nothing will do, if fall, the `app` will play the warning audio sent by the server. The `Qwen` and `vosk` model will chat with the old if the old is boring. Also the `edge` support the `app` to communicate with the old.

> [!Important]
> This repository doesn't contain the `edge` part automatically following codes and the `app` part codes. So what contains? All the server codes and the `edge` post video and audio codes, and the testing codes to see the video post to server.


> [!Attention]
> **Qwen** model runs on HUAWEI Ascend NPU. Definitely you can modify it in [Qwen.py](./server/QA/Qwen.py). The **YOLO+OpenPose** will automatically use the best device.

## QuickStart

First you must download the requirements

```bash
pip install -r requirements.txt
```

Then you need to download **Qwen** and **vosk** model.

```bash
cd download
# download Qwen
python downloadQwen.py
# download vosk
pyton downloadVosk.py
```

Now you can run the server

```bash
# start the audio model
python QA/webAudio.py
# start the predict model
python webVideo.py
```

Next you can start the edge posting part.

```bash
# Attention: you must run this on the edge, it may be Raspberry Pi.
python audioposter.py
python videoposter.py
```

Finally you can test the video. Open the [web](./client/APP/videoview.html) in your browser.
