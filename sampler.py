from transformers import AutoModelForCausalLM, AutoModel, AutoModelForSequenceClassification, AutoTokenizer, \
    PreTrainedModel
from dataclasses import dataclass
from typing import Optional, Union, Tuple
import random
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from typing import Callable, Dict, List, Optional, Tuple, Union, Any
from copy import deepcopy
from datasets import load_dataset
import os
from environment import Environment
from conversation_data import  Message,ConversationItem,ConversationManager,ConversationList
# os.environ['CUDA_VISIBLE_DEVICES'] = '2'
import re
class Sampler:
    """
    多轮对话采样器：
    - 自动构建输入batch
    - 自动生成模型回复
    - 自动交给 Environment 处理（写回对话）
    - 支持指定采样轮数
    """

    def __init__(self, model, tokenizer, env: Environment, cm: ConversationManager):
        self.model = model
        self.tokenizer = tokenizer
        self.env = env
        self.cm = cm

    @torch.no_grad()
    def sample(self, rounds: int = 1, max_new_tokens: int = 512,
               temperature=0.9, top_p=1, top_k=50):

        for step in range(rounds):
            # ===========================
            # Build batch
            # ===========================
            batch_inputs = self.cm.build_model_batch()
            if batch_inputs["input_ids"].shape[0] == 0:
                print(f"[Sampler] No more active conversations, stop at round {step}.")
                break

            # ===========================
            # Model generate
            # ===========================
            outputs = self.model.generate(
                **batch_inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k
            )

            # Remove prompt part → only model outputs
            generated_only = [
                out[len(inp):]
                for inp, out in zip(batch_inputs.input_ids, outputs)
            ]

            responses = self.tokenizer.batch_decode(generated_only, skip_special_tokens=True)

            # ===========================
            # Feed to environment
            # ===========================
            self.env.process_responses(responses)

            print(f"[Sampler] Round {step+1}/{rounds} finished.")

        print("[Sampler] Sampling completed.")

tokenizer = AutoTokenizer.from_pretrained('Qwen/Qwen2.5-0.5B-Instruct')
model = AutoModelForCausalLM.from_pretrained('Qwen/Qwen2.5-0.5B-Instruct')

# 构造对话
conv = ConversationItem(system_prompt="你是一个专业助手", tools="无")
conv.add("human", "我是你爸爸")

cl = ConversationList()
cl.add_conversation(conv)
cl.duplicate(2)

cm = ConversationManager(active=cl, tokenizer=tokenizer)
env = Environment(model, tokenizer, None, cm)

# ===== 引入采样器 =====
sampler = Sampler(model, tokenizer, env, cm)

# ===== 开始采样 =====
sampler.sample(rounds=2)