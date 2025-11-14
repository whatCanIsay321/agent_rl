import json
from typing import Optional
from pydantic import BaseModel
from conversation_data import  ConversationList,ConversationItem,Message,ConversationManager
import re
# class ToolExe:
#     def run(self,model_output):
#         try :
#             model_output ="fda"
#             {
#                 "from": "observation",
#                 "value":  f"<|im_start|>user\n<tool_response>\n{model_output}\n</tool_response><|im_end|>\n",
#             }
#         except Exception as e :
#             {
#                 "from": "observation",
#                 "value": f"<|im_start|>user\n<tool_response>\n{str(e)}\n</tool_response><|im_end|>\n",
#             }
class Environment:

    """
    一个环境类，负责：
    - 接受 user 输入 或 model 输出
    - 自动判断消息类型
    - 生成 Message 并写入 ConversationItem
    """

    def __init__(self, model,tokenizer,tool_exe,conversation_list: Con):
        self.model = model
        self.conversation_list = conversation_list
        self.tool_exe = tool_exe
    def apply_template(self):

    def build_model_inputs(self,max_length):
        dialogs = []
        for conversation in self.conversation_list:
            system_value = conversation["system_prompt"]

            tools = conversation["tools"]  # 保留转义
            tools = json.loads(tools)
            tools_str = "\n".join([json.dumps(t, ensure_ascii=False) for t in tools])
            system_prompt = f"""<|im_start|>system
            {system_value}

            # Tools

            You may call one or more functions to assist with the user query.

            You are provided with function signatures within <tools></tools> XML tags:
            <tools>
            {tools_str}
            </tools>

            For each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:
            <tool_call>
            {{"name": <function-name>, "arguments": <args-json-object>}}
            </tool_call><|im_end|>
            """
            dialog = system_prompt
            for i, turn in enumerate(conversation["conversations"]):
                role = turn["from"]
                value = turn["value"]
                if role == "human":
                    segment = f"<|im_start|>user\n{value}<|im_end|>\n"
                elif role == "gpt":
                    segment = f"<|im_start|>assistant\n{value}<|im_end|>\n"
                elif role == "observation":
                    segment = f"<|im_start|>user\n<tool_response>\n{value}\n</tool_response><|im_end|>\n"
                else:
                    raise ValueError(f"Unknown role: {role}")
                dialog += segment
            dialogs.append(dialog)

        # ===== 批量编码 =====
        enc = self.tokenizer(
            dialogs,
            add_special_tokens=False,
            max_length=max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt"
        )
        return  enc

    def has_tool_call(self,text: str) -> bool:
        return re.search(r"<tool_call>.*?</tool_call>", text, re.S) is not None

    # ===========================================================
    # 用户输入 → human 消息
    # ===========================================================


    def process_next_turn(self,responses):
        for i,item in enumerate(responses):
            if self.has_tool_call(item):
                function_call = {
                    "from": "gpt",
                    "value": item,
                }
                function_call = Message.model_validate(function_call)
                self.conversation_list.get(i).add(function_call)
                tool_response = self.tool_exe.run(item)
                observation={
                    "from": "observation",
                    "value": tool_response,
                }
                observation=Message.model_validate(observation)
                self.conversation_list.get(i).add(observation)
            else:
                data = {
                    "from": "gpt",
                    "value": item,
                }
                msg = Message.model_validate(data)
                self.conversation_list.get(i).add(msg)
    def reward(self,conversation_list):
        return [1]*len(conversation_list)


