from pydantic import BaseModel, Field
from typing import Literal, Optional, Any, List
import copy
import torch
from transformers import BatchEncoding


# ============================================================
# Message
# ============================================================
class Message(BaseModel):
    from_: Literal["human", "gpt", "observation"] = Field(alias="from")
    value: Any

    def export(self):
        return self.model_dump(by_alias=True)


# ============================================================
# ConversationItem
# ============================================================
class ConversationItem(BaseModel):
    system_prompt: str = ""
    tools: str = ""
    conversations: List[Message] = Field(default_factory=list)

    # for RL training reward (optional)
    answer: Optional[Any] = None
    def add_message(self, message: Message | dict):
        if isinstance(message, dict):
            message = Message.model_validate(message)
        self.conversations.append(message)

    def export(self):
        return {
            "system_prompt": self.system_prompt,
            "tools": self.tools,
            "conversations": [msg.export() for msg in self.conversations],
            "answer": self.answer,
        }


# ============================================================
# ConversationList
# ============================================================
class ConversationList(BaseModel):
    items: List[ConversationItem] = Field(default_factory=list)

    def add_conversation(self, conversation):
        if isinstance(conversation, dict):
            conversation = ConversationItem.model_validate(conversation)
        self.items.append(conversation)

    def pop(self, index):
        return self.items.pop(index)

    def clone(self):
        return copy.deepcopy(self)

    def duplicate(self, times: int):
        if times <= 0:
            return
        extra = []
        for conv in self.items:
            extra.extend(conv.clone() for _ in range(times))
        self.items.extend(extra)


# ============================================================
# ConversationManager (GRPO ONLY)
# ============================================================
class ConversationManager(BaseModel):
    active: ConversationList = Field(default_factory=ConversationList)
    finished: ConversationList = Field(default_factory=ConversationList)
    tokenizer: Any = None

    # ============================================================
    # System prompt builder
    # ============================================================
    def build_system_prompt(self, system_value, tools_str):
        return f"""<|im_start|>system
{system_value}

# Tools

You may call one or more functions to assist with the user query.

You are provided with function signatures within <tools></tools> XML tags:
<tools>
{tools_str}
</tools>

For each function call, return:
<tool_call>{{"name": <function-name>, "arguments": <args-json-object>}}</tool_call>
<|im_end|>
"""

    # ============================================================
    # Turn formatter
    # ============================================================
    def format_turn(self, turn):
        role = turn["from"]
        val = turn["value"]

        if role == "human":
            return f"<|im_start|>user\n{val}<|im_end|>\n"

        if role == "gpt":
            return f"<|im_start|>assistant\n{val}<|im_end|>\n"

        if role == "function_call":
            return f"<|im_start|>assistant\n{val}<|im_end|>\n"

        if role == "observation":
            return f"<|im_start|>user\n<tool_response>\n{val}\n</tool_response><|im_end|>\n"

        raise ValueError(f"unknown role: {role}")

    # ============================================================
    # 拼接 + tokenizer 编码（无 padding）
    # 返回：enc, spans (assistant token spans)
    # ============================================================
    def encode_dialog(self, conv_dict):
        system = self.build_system_prompt(conv_dict["system_prompt"], conv_dict["tools"])

        dialog = system
        spans = []

        for turn in conv_dict["conversations"]:
            seg = self.format_turn(turn)
            start = len(dialog)
            dialog += seg
            end = len(dialog)

            if turn["from"] == "gpt":
                spans.append((start, end))

        enc = self.tokenizer(
            dialog,
            return_offsets_mapping=True,
            add_special_tokens=False,
            truncation=False,
        )
        return enc, spans

    # ============================================================
    # 左 padding 工具
    # ============================================================
    def left_pad(self, ids, mask, other=None, max_length=2048):
        pad_id = self.tokenizer.pad_token_id or self.tokenizer.eos_token_id
        L = len(ids)

        if L > max_length:
            ids = ids[-max_length:]
            mask = mask[-max_length:]
            if other is not None:
                other = other[-max_length:]
        else:
            pad = max_length - L
            ids = [pad_id] * pad + ids
            mask = [0] * pad + mask
            if other is not None:
                other = [0] * pad + other  # padding 永远是 0

        out = {
            "input_ids": torch.tensor(ids, dtype=torch.long),
            "attention_mask": torch.tensor(mask, dtype=torch.long),
        }
        if other is not None:
            out["other"] = torch.tensor(other, dtype=torch.long)

        return BatchEncoding(out)

    # ============================================================
    # ① GRPO Rollout 阶段：仅生成模型输入
    # ============================================================
    def build_model_batch(self, max_length=2048):
        batch = []
        remove = []

        for idx, conv in enumerate(self.active.items):
            enc, _ = self.encode_dialog(conv.export())
            ids = enc["input_ids"]

            if len(ids) >= max_length:
                remove.append(idx)
                continue

            padded = self.left_pad(ids, enc["attention_mask"], other=None, max_length=max_length)
            batch.append(padded)

        for i in reversed(remove):
            self.finished.add_conversation(self.active.pop(i))

        if not batch:
            return BatchEncoding({
                "input_ids": torch.empty(0),
                "attention_mask": torch.empty(0),
            })

        return BatchEncoding({
            "input_ids": torch.stack([x["input_ids"] for x in batch]),
            "attention_mask": torch.stack([x["attention_mask"] for x in batch]),
        })

    # ============================================================
    # ② GRPO 训练阶段：输出 action_mask (0/1)
    # ============================================================
    def build_grpo_sample(self, conv_dict, max_length=2048):
        """
        构建单条 GRPO 样本（单条对话 → tokenized → action_mask → 左padding）
        conv_dict 必须是 conv.export() 的结果
        """

        # === 编码对话，得到 spans (哪些token是助手输出) ===
        enc, spans = self.encode_dialog(conv_dict)

        ids = enc["input_ids"]
        mask = enc["attention_mask"]
        offsets = enc["offset_mapping"]

        # === assistant 的 token → 1，其余 → 0 ===
        action_mask = []
        for (tok_start, _), tid in zip(offsets, ids):
            in_span = any(a <= tok_start < b for a, b in spans)
            action_mask.append(1 if in_span else 0)

        # === 做左 padding：input_ids / mask / action_mask 全部 pad ===
        padded = self.left_pad(
            ids,
            mask,
            other=action_mask,
            max_length=max_length
        )

        # === 返回为 BatchEncoding，保持和你原来一致 ===
        return BatchEncoding({
            "input_ids": padded["input_ids"],
            "attention_mask": padded["attention_mask"],
            "action_mask": padded["other"],
        })

    def build_grpo_batch(self, max_length=2048):
        """
        使用 export_all() 获取 active + finish 全部对话
        构建一个批次的 GRPO BatchEncoding
        """

        # 1. 获取所有对话（active + finish）
        conv_dicts = self.export_all()

        # 2. 构建单条 GRPO 样本
        samples = [
            self.build_grpo_sample(conv_dict, max_length)
            for conv_dict in conv_dicts
        ]

        # 3. 堆叠为 BatchEncoding
        return BatchEncoding({
            "input_ids": torch.stack([x["input_ids"] for x in samples]),
            "attention_mask": torch.stack([x["attention_mask"] for x in samples]),
            "action_mask": torch.stack([x["action_mask"] for x in samples]),
        })

    def export_all(self):
        """导出 active 和 finish 中所有对话为列表"""
        all_items = []

        if hasattr(self, "active") and hasattr(self.active, "items"):
            all_items.extend([conv.export() for conv in self.active.items])

        if hasattr(self, "finish") and hasattr(self.finish, "items"):
            all_items.extend([conv.export() for conv in self.finish.items])

        return all_items


if __name__ == "__main__":
    from transformers import AutoTokenizer, AutoModelForCausalLM

    # -------------------------
    # 加载 Qwen tokenizer & model
    # -------------------------
    model_name = "Qwen/Qwen2.5-0.5B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name).cuda()

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

    # 多轮
    conv.add_message("human", "你好，我有一个数学问题。")
    conv.add_message("gpt", "你好，请问是什么问题？")
    conv.add_message("human", "123 + 456 等于多少？")

    # 添加进 active 列表
    cm.active.add_conversation(conv)

    print("=== 原始多轮对话 ===")
    for m in conv.conversations:
        print(m.from_, ":", m.value)

    # ============================================================
    # ① 构造 rollout 输入（model 输入）
    # ============================================================
    print("\n=== 构造 rollout batch ===")
    batch = cm.build_model_batch(max_length=256)

    print("input_ids.shape =", batch["input_ids"].shape)
    print("attention_mask.shape =", batch["attention_mask"].shape)

    # ============================================================
    # ② 利用模型 generate（生成助手回复）
    # ============================================================
    print("\n=== 模型生成回复 ===")
    with torch.no_grad():
        gen = model.generate(
            input_ids=batch["input_ids"].cuda(),
            attention_mask=batch["attention_mask"].cuda(),
            max_new_tokens=100
        )

    # 取新生成的部分（右边）
    input_len = batch["input_ids"].shape[1]
    new_tokens = gen[0][input_len:]
    reply = tokenizer.decode(new_tokens, skip_special_tokens=True)

    print("模型回复：", reply)

    # 将回复加入对话
    conv.add("gpt", reply)

    # ============================================================
    # ③ 构建 GRPO 训练 batch（带 action_mask=0/1）
    # ============================================================
    print("\n=== 构造 GRPO 训练 batch ===")
    grpo_batch = cm.build_grpo_batch(max_length=512)

    print("input_ids.shape =", grpo_batch["input_ids"].shape)
    print("action_mask.shape =", grpo_batch["action_mask"].shape)

    print("\n=== action_mask ===")
    print(grpo_batch["action_mask"])
