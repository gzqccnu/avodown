# avodown
# Copyright (c) 2025 gzqccnu 
# 
# This program is released under the terms of the Apache License.
# See https://opensource.org/licenses/Apache for more information.
# 
# Project homepage: https://github.com/gzqccnu/avodown
# Description: Using models to avoid the old fall down

from transformers import AutoModelForCausalLM, AutoTokenizer

# 指定模型名称
model_name = "Qwen/Qwen1.5-1.8B-Chat"
save_path = "./qwen_1.5b_chat_model"  # 本地保存路径

# 加载模型和分词器
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",
    trust_remote_code=True
)

print(type(model))  # 应为 QwenForCausalLM 或类似
print(hasattr(model, "chat"))  # 应为 True

# 保存到本地
tokenizer.save_pretrained(save_path)
model.save_pretrained(save_path)

print(f"模型已成功下载到 {save_path}")
