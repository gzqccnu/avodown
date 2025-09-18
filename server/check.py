from transformers import AutoModelForCausalLM, AutoTokenizer

model_path = "./qwen_1.5b_chat_model"

tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True)

print("✅ 成功加载模型类型：", type(model))
print("是否有 chat 方法？", hasattr(model, "chat"))
