# avodown
# Copyright (c) 2025 gzqccnu <gzqccnu@gmail.com>
# 
# This program is released under the terms of the Apache License.
# See https://opensource.org/licenses/Apache for more information.
# 
# Project homepage: https://github.com/gzqccnu/avodown
# Description: Using models to avoid the old fall down

import os
import torch
import torch_npu
from transformers import AutoTokenizer, AutoModelForCausalLM

class QwenServer:
    def __init__(self, model_path: str, device: str = 'npu:1'):
        # ——— NPU 环境启动 ———
        if device.startswith('npu'):
            os.environ['NPU_VISIBLE_DEVICES'] = device.split(':')[-1]
            torch.npu.set_compile_mode(jit_compile=True)
            torch.npu.config.allow_internal_format = True
        self.device = torch.device(device)

        # ——— 加载模型 & Tokenizer & 半精度 ———
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path, trust_remote_code=True
        ).half().to(self.device)
        self.model.eval()

        # ——— 预热一次，关闭后续编译 ———
        dummy_ids = torch.zeros((1,1), dtype=torch.long, device=self.device)
        _ = self.model.generate(dummy_ids, max_new_tokens=1)
        # 预热完成后关闭 JIT 编译（避免动态重编译）
        torch.npu.set_compile_mode(jit_compile=False)

        # ——— 拆分系统 Prompt，预先 tokenize ———
        self.system_prompt = "你是一个智能语音助手，回答要简洁自然。"
        sys_toks = self.tokenizer(
            f"<|im_start|>system\n{self.system_prompt}<|im_end|>\n",
            return_tensors="pt"
        )
        self.sys_input_ids = sys_toks["input_ids"].to(self.device)
        self.sys_attn_mask = sys_toks["attention_mask"].to(self.device).to(torch.int32)

        # 对话历史（只保存文本）
        self.history = []

    def chat(self, text: str) -> str:
        # 1) Append 用户输入
        self.history.append({"role": "user", "content": text})

        # 2) 构建增量 Prompt：只 tokenize 最新两轮
        #    system + history_texts + assistant prompt
        hist = ""
        for turn in self.history:
            hist += f"<|im_start|>{turn['role']}\n{turn['content']}<|im_end|>\n"
        full_prompt = self.sys_input_ids, self.sys_attn_mask  # 占位

        # 拼接：先 system prompt tokens，再 tokenize history 文本
        history_toks = self.tokenizer(
            hist + "<|im_start|>assistant\n",
            return_tensors="pt",
            padding=False,
            truncation=False
        ).to(self.device)
        # 合并 input_ids & attention_mask
        """
        origin
        """
        input_ids = torch.cat([self.sys_input_ids, history_toks["input_ids"]], dim=1)
        attn_mask = torch.cat([self.sys_attn_mask, history_toks["attention_mask"].to(torch.int32)], dim=1)

        # 3) 生成：启用 use_cache 提升多 token 生成效率
        output_ids = self.model.generate(
            input_ids=input_ids,
            attention_mask=attn_mask,
            max_new_tokens=256,
            do_sample=True,
            temperature=0.7,
            eos_token_id=self.tokenizer.convert_tokens_to_ids("<|im_end|>"),
            use_cache=True,           # 激活 KV cache
            pad_token_id=self.tokenizer.eos_token_id
        )

        # 4) 解码 & 更新历史
        text_out = self.tokenizer.decode(output_ids[0][input_ids.shape[-1]:], skip_special_tokens=True)
        reply = text_out.split("<|im_start|>assistant\n")[-1].split("<|im_end|>")[0].strip()
        self.history.append({"role": "assistant", "content": reply})
        return reply
        """
        origin
        """
