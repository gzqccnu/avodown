# avodown
# Copyright (c) 2025 gzqccnu 
#
# This program is released under the terms of the Apache License.
# See https://opensource.org/licenses/Apache for more information.
#
# Project homepage: https://github.com/gzqccnu/avodown
# Description: Using models to avoid the old fall down

from transformers import AutoModelForCausalLM, AutoTokenizer

model_path = "./qwen_1.5b_chat_model"

tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True)

print("✅ 成功加载模型类型：", type(model))
print("是否有 chat 方法？", hasattr(model, "chat"))
