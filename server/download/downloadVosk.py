from vosk import Model

model = Model(r"./vosk-model-cn-0.22") # 这里替换为你想要保存模型的路径
# 或者直接指定语言，让其自动下载（前提是网络通畅且能访问官网）
model = Model(lang="cn") 
