import re
from conversation_data import *
class Environment:
    """
    环境类：
    - 接受模型 response（batch）
    - 自动识别 tool_call
    - 自动写入 ConversationManager.active.items[i]
    """

    def __init__(self, model, tokenizer, tool_exe, conversation_manager):
        self.model = model
        self.tokenizer = tokenizer
        self.tool_exe = tool_exe
        self.cm = conversation_manager   # ← 传入的是 ConversationManager

    def has_tool_call(self, text: str) -> bool:
        return re.search(r"<tool_call>.*?</tool_call>", text, re.S) is not None

    # ===========================================================
    # 核心：处理模型 batch 输出
    # ===========================================================
    def process_responses(self, responses):
        """
        responses: list[str]
        自动写入到 cm.active.items[i].conversations
        """

        for i, text in enumerate(responses):

            # 如果 active 不足以索引（可能有些被 finish），跳过
            if i >= len(self.cm.active.items):
                continue

            conv: ConversationItem = self.cm.active.items[i]

            # ===== case 1: tool_call =====
            if self.has_tool_call(text):

                # assistant 原始输出
                conv.add("gpt", text)

                # 工具执行
                try:
                    tool_result = self.tool_exe.run(text)
                except Exception as e:
                    tool_result = str(e)

                # observation
                conv.add("observation", tool_result)

            # ===== case 2: 普通助手回复 =====
            else:
                conv.add("gpt", text)

    # 可自定义 reward
    def reward(self):
        return [1] * len(self.cm.active.items)
