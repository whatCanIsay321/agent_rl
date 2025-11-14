from pydantic import BaseModel, Field
from typing import Literal, Optional, Any, List
import copy

# ==========================
# Message
# ==========================
class Message(BaseModel):
    from_: Literal["human", "gpt", "observation"] = Field(alias="from")
    value: Any

    def export(self):
        return self.model_dump(by_alias=True)


# ==========================
# ConversationItem
# ==========================
class ConversationItem(BaseModel):
    system_prompt: str = ""
    tools: str = ""
    conversations: List[Message] = Field(default_factory=list)

    def add(self, role: Literal["human", "gpt", "observation"], value: Any):
        msg = Message.model_validate({"from": role, "value": value})
        self.conversations.append(msg)

    def get_message(self, index: int) -> Message:
        if not self.conversations:
            raise IndexError("Conversation is empty")
        if index < -len(self.conversations) or index >= len(self.conversations):
            raise IndexError("Index out of range")
        return self.conversations[index]

    def clone(self):
        return copy.deepcopy(self)

    def export(self):
        return {
            "system_prompt": self.system_prompt,
            "tools": self.tools,
            "conversations": [msg.export() for msg in self.conversations]
        }


# ==========================
# ConversationList
# ==========================
class ConversationList(BaseModel):
    items: List[ConversationItem] = Field(default_factory=list)

    def add_conversation(self, conversation: ConversationItem | dict):
        if isinstance(conversation, dict):
            conversation = ConversationItem.model_validate(conversation)
        self.items.append(conversation)

    def clone(self):
        return copy.deepcopy(self)

    def duplicate(self, times: int):
        if times <= 0:
            return
        new_items = []
        for conv in self.items:
            for _ in range(times):
                new_items.append(conv.clone())
        self.items.extend(new_items)

    def get(self, index: int) -> ConversationItem:
        if not self.items:
            raise IndexError("ConversationList is empty")
        if index < -len(self.items) or index >= len(self.items):
            raise IndexError("Index out of range")
        return self.items[index]

    def export_all(self):
        return [conv.export() for conv in self.items]


# ==========================
# ConversationManager
# ==========================
class ConversationManager(BaseModel):
    active: ConversationList = Field(default_factory=ConversationList)
    finished: ConversationList = Field(default_factory=ConversationList)

    def add(self, conv: ConversationItem):
        self.active.add_conversation(conv)

    def finish(self, index: int):
        conv = self.active.items.pop(index)
        self.finished.add_conversation(conv)

    def finish_many(self, indices: List[int]):
        for idx in sorted(indices, reverse=True):
            self.finish(idx)

    def all(self):
        return self.active.items + self.finished.items


# =====================================================================
# ======================  一次性完整使用示例  ===========================
# =====================================================================
if __name__ == "__main__":

    print("\n=====【1. 构建 Message 示例】=====")
    msg1 = Message.model_validate({"from": "human", "value": "你好，我是用户"})
    msg2 = Message.model_validate({"from": "gpt", "value": "你好，我能为你做什么？"})
    print(msg1.export())
    print(msg2.export())

    print("\n=====【2. 构建 ConversationItem 示例】=====")
    conv = ConversationItem(
        system_prompt="你是一个专业助手",
        tools="无",
        conversations=[msg1, msg2]
    )
    conv.add("human", "请问今天天气如何？")
    conv.add("gpt", "今天天气晴朗，气温 18-25 度。")
    print(conv.export())

    print("\n=====【3. 创建 ConversationList + duplicate 示例】=====")
    cl = ConversationList()
    cl.add_conversation(conv)

    # duplicate 2 次 → 共 3 条
    cl.duplicate(times=2)
    print(f"ConversationList 当前数量: {len(cl.items)}")  # 3

    print("\n=====【4. 使用 get() 和 get_message() 示例】=====")
    first_conv = cl.get(0)
    last_msg = first_conv.get_message(-1)
    print("第一条对话最后一句:", last_msg.export())

    print("\n=====【5. ConversationManager 管理 active/finished】=====")
    cm = ConversationManager()
    cm.add(conv.clone())
    cm.add(conv.clone())
    cm.add(conv.clone())
    print("active 数量:", len(cm.active.items))   # 3

    cm.finish(1)  # 把第二条移到 finished
    print("active 数量:", len(cm.active.items))   # 2
    print("finished 数量:", len(cm.finished.items))  # 1

    cm.finish_many([0, 1])   # 把剩余两个也移动
    print("active:", len(cm.active.items))  # 0
    print("finished:", len(cm.finished.items))  # 3

    print("\n=====【6. RL Rollout 演示：生成多条轨迹】=====")

    # 初始 query
    init_conv = ConversationItem(
        system_prompt="你是一个助手",
        tools="无"
    )
    init_conv.add("human", "请介绍一下你自己")

    # 放入列表并复制 2 次 → 共 3 条 rollout
    rollouts = ConversationList()
    rollouts.add_conversation(init_conv)
    rollouts.duplicate(times=2)

    # 模拟每条 rollout 得到不同回答
    rollout_answers = ["我是助手 A", "我是助手 B", "我是助手 C"]
    for idx, (conv_item, ans) in enumerate(zip(rollouts.items, rollout_answers)):
        conv_item.add("gpt", ans)

    # 打印结果
    for idx, conv_item in enumerate(rollouts.items):
        print(f"\n----- Rollout {idx} -----")
        print(conv_item.export())

    print("\n=====【全部结束】=====")
