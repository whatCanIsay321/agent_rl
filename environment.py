import re
from conversation_data import *
from ToolExe import *
import json
import re
from typing import Generic, TypeVar, Type, Any
from pydantic import BaseModel, RootModel, ValidationError
from transformers import AutoTokenizer


class Environment:
    """
    环境类：
    - 接受模型 response（batch）
    - 自动识别 tool_call
    - 自动写入 ConversationManager.active.items[i]
    """

    def __init__(self,tokenizer, tool_exe, conversation_manager,model=None, ):
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
            # if i >= len(self.cm.active.items):
            #     continue

            conv: ConversationItem = self.cm.active.items[i]

            # ===== case 1: tool_call =====
            if self.has_tool_call(text):
                # assistant 原始输出
                conv.add_message({"from": "function_call", "value": text})
                # 工具执行
                try:
                    tool_result = self.tool_exe.run(text)
                except Exception as e:
                    tool_result = str(e)

                # observation
                conv.add_message({"from": "observation", "value": tool_result})

            # ===== case 2: 普通助手回复 =====
            else:
                conv.add_message({"from": "gpt", "value": text})

    # 可自定义 reward
    def reward(self):

        conv_dicts = self.cm.export_all()



        return [1] * len(self.cm.export_all())

if __name__ == "__main__":
    class IntroInput(BaseModel):
        name: str
        lang: str = "zh"


    class IntroOutput(RootModel[str]):
        """你好"""
        pass


    class IntroTool(ToolBase[IntroInput, IntroOutput]):
        name = "自我介绍"
        description = "返回自我介绍文本"

        Input = IntroInput
        Output = IntroOutput

        def run(self, data: IntroInput):
            if data.lang == "zh":
                return f"你好，{data.name}！我是天智AI。"
            return f"Hello, {data.name}! I'm Tianzhi AI."


    class ListUsersInput(BaseModel):
        limit: int = 5


    from typing import List


    class ListUsersOutput(RootModel[List[str]]):
        """返回用户列表"""
        pass


    class ListUsersTool(ToolBase[ListUsersInput, ListUsersOutput]):
        name = "获取用户列表"
        description = "返回一个字符串列表，表示用户昵称"

        Input = ListUsersInput
        Output = ListUsersOutput

        def run(self, data: ListUsersInput):
            # 简单模拟：返回 limit 个用户名字
            users = [f"用户{i + 1}" for i in range(data.limit)]
            return users  # RootModel[List[str]] 的正确返回方式


    tool_exe = ToolExe()
    tool_exe.register(IntroTool())
    tool_exe.register(ListUsersTool())
    model_name = "Qwen/Qwen2.5-0.5B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    # model = AutoModelForCausalLM.from_pretrained(model_name).cuda()
    # model.eval()

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

    env = Environment(tokenizer,tool_exe,cm)
    rewar = env.reward()



    #
    # print("=== 工具 JSON Schema ===")
    # x = tool_exe.export_schemas()
    # print(tool_exe.export_schemas())
    #
    # model_output = """
    # <tool_call>
    # {"name": "自我介绍", "arguments": {"name": "小明", "lang": "zh"}}
    # </tool_call>
    # """
    #
    # print("\n=== 工具执行结果 ===")
    # print(tool_exe.run(model_output))