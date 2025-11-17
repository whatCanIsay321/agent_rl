from conversation_data import *
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
from conversation_data import  Message,ConversationItem,ConversationManager,ConversationList
# os.environ['CUDA_VISIBLE_DEVICES'] = '2'
import re
from  environment import Environment
def generate_samples():
    from transformers import AutoTokenizer, AutoModelForCausalLM
    model_name = "Qwen/Qwen2.5-0.5B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name).cuda()
    model.eval()

    # -------------------------
    # 初始化 ConversationManager
    # -------------------------
    cm = ConversationManager(tokenizer=tokenizer)

    # -------------------------
    # 构造一个多轮对话 ConversationItem
    # -------------------------
    conv = ConversationItem(
        system_prompt="你是一个数学推理助手。",
        tools="[]"
    )
    conv1 = ConversationItem(
        system_prompt="nml",
        tools="[]"
    )

    # 多轮
    conv.add_message({"from":"gpt", "value":"你好，我有一个数学问题。"})
    # conv.add("gpt", "你好，请问是什么问题？")
    # conv.add("human", "123 + 456 等于多少？")

    # 添加进 active 列表
    cm.active.add_conversation(conv)
    cm.active.add_conversation(conv1)

    print("=== 原始多轮对话 ===")
    # for m in conv.conversations:
    #     print(m.from_, ":", m.value)

    # ============================================================
    # ① 构造 rollout 输入（model 输入）
    # ============================================================
    print("\n=== 构造 rollout batch ===")
    # batch = cm.build_model_batch(max_length=256).to("cuda")
    batch = cm.build_grpo_batch(max_length=256).to("cuda")
    #
    # print("input_ids.shape =", batch["input_ids"].shape)
    # print("attention_mask.shape =", batch["attention_mask"].shape)
    #
    # # ============================================================
    # # ② 利用模型 generate（生成助手回复）
    # # ============================================================
    # print("\n=== 模型生成回复 ===")
    with torch.no_grad():
        generated_ids = model.generate(**batch,
            max_new_tokens=100
        )

    # 取新生成的部分（右边）
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(batch.input_ids, generated_ids)
    ]

    reply = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)

    print("模型回复：", reply)
    active_list = cm.active
    for i, conv in enumerate(active_list.items):
        conv.add_message({"from":"gpt", "value":reply[i]})
    将回复加入对话

generate_samples()

def get_action_log_probs(model, input_ids, attention_mask, action_mask):
    out = model(input_ids, attention_mask=attention_mask)
    logits = out.logits                          # (B, L, V)

    log_probs = F.log_softmax(logits, dim=-1)    # (B, L, V)
    target = input_ids.unsqueeze(-1)             # (B, L, 1)

    token_log_probs = log_probs.gather(-1, target).squeeze(-1)  # (B, L)

    action_log_probs = token_log_probs * action_mask

    return action_log_probs
model_name = "Qwen/Qwen2.5-0.5B-Instruct"
model = AutoModelForCausalLM.from_pretrained(model_name).cuda()
def generate_exprience():

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    model.eval()
    env = Environment()
    # -------------------------
    # 初始化 ConversationManager
    # -------------------------
    cm = ConversationManager(tokenizer=tokenizer)

    # -------------------------
    # 构造一个多轮对话 ConversationItem
    # -------------------------
    conv = ConversationItem(
        system_prompt="你是一个数学推理助手。",
        tools="[]"
    )
    conv1 = ConversationItem(
        system_prompt="nml",
        tools="[]"
    )

    # 多轮
    conv.add_message({"from":"human", "value":"你好，我有一个数学问题。"})

    # 添加进 active 列表
    cm.active.add_conversation(conv)
    cm.active.add_conversation(conv1)
    batch = cm.build_grpo_batch().to("cuda")
    with torch.no_grad():
            prompt_response_ids = batch.input_ids
            attention_mask = batch.attention_mask

            # 计算策略模型输出token的概率
            old_action_log_probs = get_action_log_probs(model, prompt_response_ids, attention_mask,
                                                             num_actions)
            batch_old_action_log_probs.append(old_action_log_probs)

           # 是否使用参考模型
            if ref_model:
                # 计算参考模型输出token的概率
                ref_action_log_probs = get_action_log_probs(ref_model, prompt_response_ids,
                                                                 attention_mask, num_actions)
                batch_ref_action_log_probs.append(ref_action_log_probs)

            # 存储各个奖励函数在一个group内各个响应的奖励
            rewards_per_func = torch.zeros(len(reward_funcs), args.num_generations,
                                           device=args.device)

        advantages = env.reward()

        return {
            "prompt_response_ids": torch.cat(batch_prompt_response_ids, dim=0),
            "attention_mask": torch.cat(batch_attention_mask, dim=0),
            "action_mask": torch.cat(batch_action_mask, dim=0),
            "old_action_log_probs": torch.cat(batch_old_action_log_probs, dim=0),
            "ref_action_log_probs": torch.cat(batch_ref_action_log_probs, dim=0) if ref_model else None,
            "advantages": torch.cat(batch_advantages, dim=0),
        }


#
def compute_loss( model, inputs):

    prompt_response_ids = inputs['prompt_response_ids']   # (B, L)
    attention_mask      = inputs['attention_mask']        # (B, L)
    action_mask         = inputs['action_mask']           # (B, L)

    # === 1. 计算当前 policy πθ 的 log π(a|s) ===
    # 输出 shape = (B, L)，非 action token 为 0
    action_log_probs = get_action_log_probs(
        model,
        prompt_response_ids,
        attention_mask,
        action_mask
    )

    # === 2. 计算 KL 项（如果 beta != 0）===
    if args.beta != 0.0:
        ref_action_log_probs = inputs['ref_action_log_probs']  # (B, L)
        # 这里是 token-level KL term： log ref - log policy
        log_ratio = (ref_action_log_probs - action_log_probs) * action_mask
        k3 = (log_ratio.exp() - 1 - log_ratio) * action_mask   # GRPO 的 KL 结构

    # === 3. 计算 advantages ===
    # advantages shape = (B,)
    advantages = inputs['advantages']       # 序列级

    # === 4. old log prob（用于 PPO clipping）===
    if args.num_iterations > 1:
        old_action_log_probs = inputs["old_action_log_probs"]  # (B, L)
    else:
        old_action_log_probs = action_log_probs.detach()       # (B, L)

    # === 5. PPO 的 ratio：exp(log πθ − log πθ_old) ===
    # 依然是 token-level
    ratio = torch.exp(action_log_probs - old_action_log_probs)  # (B, L)

    # === mask 掉无效 token ===
    ratio = ratio * action_mask

    # === 6. 计算 PPO clip 部分 ===
    adv = advantages.unsqueeze(1)           # (B, 1) → token-level broadcast

    loss1 = ratio * adv                     # (B, L)
    loss2 = torch.clamp(
        ratio,
        1 - args.clip_eps,
        1 + args.clip_eps
    ) * adv

    per_token_loss = -torch.min(loss1, loss2)   # (B, L)
    per_token_loss = per_token_loss * action_mask  # mask 掉非 action token

    # === 7. 加 GRPO KL 正则项 ===
    if args.beta != 0.0:
        per_token_loss = per_token_loss + args.beta * k3

    # === 8. 序列归一化 loss ===
    loss = per_token_loss.sum(dim=1) / action_mask.sum(dim=1)   # (B,)
    loss = loss.mean()  # 最终 batch loss

    return loss
x = generate_exprience()
