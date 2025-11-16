
import json
from transformers import  AutoTokenizer



tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct",local_files_only=True)
#
conversation = {"conversations": [{"from": "human", "value": "你是？"},
                                  {"from": "function_call",
"value": "准确和及时的信息，并通过各种方式帮助用户进行有效的沟通。请告诉我有什么可以帮助您的呢？{\"name\": \"自我介绍\", \"""arguments\": {\"content\": \"我是天智AI，由天合富家训练的人工智能助手。我的目标是为用户提供有用、准确和及时的信息，并通过各种方式帮助用户进行有效的沟通。请告诉我有什么可以帮助您的呢？\"}}"},
                                  {
                                      "from": "observation",
                                      "value": "{\"invoice_id\": \"INV12345\", \"customer_name\": \"约翰·多伊\", \"items\": [{\"name\": \"苹果\", \"quantity\": 2, \"price\": 1, \"total\": 2}, {\"name\": \"香蕉\", \"quantity\": 3, \"price\": 0.5, \"total\": 1.5}], \"total\": 3.5, \"status\": \"生成\"}"
                                  },
                                  {
                                      "from": "gpt",
                                      "value": "发票已成功生成。发票编号为INV12345。约翰·多伊的总金额为$3.5。发票包含2个苹果，总金额为$2，以及3根香蕉，总金额为$1.5。"
                                  }
                                  ],
"system_prompt": "You are TianZhi, created by THFJ. You are a helpful assistant", "tools": "[{\"type\": \"function\", \"function\": {\"name\": \"自我介绍\", \"description\": \"进行自我介绍回答的函数\", \"parameters\": {\"properties\": {\"content\": {\"default\": \"我是天智AI，请问需要帮助吗？\", \"description\": \"用来回复用户的回答内容\", \"type\": \"string\"}}, \"type\": \"object\"}}}, {\"type\": \"function\", \"function\": {\"name\": \"质量\", \"description\": \"处理勘探订单、图纸与安装服务相关流程的系统，亦称“质量系统”或“全流程系统”。\", \"parameters\": {\"properties\": {}, \"type\": \"object\"}}}, {\"type\": \"function\", \"function\": {\"name\": \"进件\", \"description\": \"处理非全款业务流程的系统，亦称“金融系统”或“进件系统”。\", \"parameters\": {\"properties\": {}, \"type\": \"object\"}}}, {\"type\": \"function\", \"function\": "
                                                                                                                                                                                                                                                                                                                                                       "{\"name\": \"闲聊\", \"description\": \"处理日常对话的系统。\", \"parameters\": {\"properties\": {}, \"type\": \"object\"}}}]"}

system_value = conversation["system_prompt"]
tools_str = conversation["tools"]  # 保留转义

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
</tool_call>
<|im_end|>
"""

# === 拼接对话内容 ===
dialog = system_prompt
spans = []

for i, turn in enumerate(conversation["conversations"]):
    role = turn["from"]
    value = turn["value"]
    if role == "human":
        segment = f"<|im_start|>user\n{value}<|im_end|>\n"
    elif role == "function_call":
        segment = f"<|im_start|>assistant\n{value}<|im_end|>\n"
    elif role == "observation":
        segment = f"<|im_start|>user\n<tool_response>\n{value}\n</tool_response><|im_end|>\n"
    elif role == "gpt":
        segment = f"<|im_start|>assistant\n{value}<|im_end|>\n"
    else:
        raise ValueError(f"Unknown role: {role}")

    start = len(dialog)
    dialog += segment
    end = len(dialog)
    spans.append((start, end))

# === 编码并生成 input_ids 和 label_ids ===
enc = tokenizer(dialog, return_offsets_mapping=True, add_special_tokens=False)
attention_mask = enc["attention_mask"]
input_ids = enc["input_ids"]
offsets = enc["offset_mapping"]
label_ids = []
for (token_start, _), token_id in zip(offsets, input_ids):
    keep = False
    for i, (span_start, span_end) in enumerate(spans):
        if i % 2 == 1 and span_start <= token_start < span_end:
            keep = True
            break
    label_ids.append(1 if keep else 0)



# === 输出 ===
print("=== 拼接后的 dialog ===")
print(dialog)

# print("\n=== input_ids (text) ===")
# print(tokenizer.decode(input_ids))

print("\n=== label_ids (with -100) ===")
print(label_ids)

filtered_token_ids = [token_id for token_id in label_ids if token_id != -100]

# 解码
decoded_output = tokenizer.decode(filtered_token_ids)

print("\n=== 解码后的非 -100 部分（即训练目标）===")
print(decoded_output)
#

