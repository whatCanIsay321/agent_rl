
from langgraph.constants import START,END
from langgraph.graph import StateGraph
from transformers import AutoTokenizer,AutoModelForCausalLM
from typing import TypedDict, Any
model_name= "Qwen/Qwen2.5-0.5B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=True)
model  = AutoModelForCausalLM.from_pretrained(model_name,local_files_only=True)
class agent:
    def __int__(self,tokenizer,model):
        se

    def build_model_inputs(self, max_length):
        dialogs = []

        # ===== 构建所有对话的文本 =====
        for conversation in self.conversation_list.items:  # 注意，conversation_list.items 才是对象
            system_value = conversation.system_prompt
            tools = conversation.tools
            tools = json.loads(tools) if isinstance(tools, str) else tools
            tools_str = "\n".join(json.dumps(t, ensure_ascii=False) for t in tools)

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

            # ===== 多轮对话拼接 =====
            for turn in conversation.conversations:
                role = turn.from_
                value = turn.value

                if role == "human":
                    dialog += f"<|im_start|>user\n{value}<|im_end|>\n"

                elif role == "gpt":
                    dialog += f"<|im_start|>assistant\n{value}<|im_end|>\n"

                elif role == "observation":
                    dialog += (
                        f"<|im_start|>user\n<tool_response>\n{value}\n"
                        f"</tool_response><|im_end|>\n"
                    )

            dialogs.append(dialog)

        # =============================================================================
        # 手动编码 + 手动 padding
        # =============================================================================

        # 1. 逐条 encode（不 pad）
        encoded_list = [
            self.tokenizer(
                d,
                add_special_tokens=False,
                truncation=True,  # 截断到 max_length
                max_length=max_length
            )["input_ids"]
            for d in dialogs
        ]

        # 2. 找到 batch 内最长 sequence
        max_seq_len = max(len(ids) for ids in encoded_list)

        # 3. 手动 padding input_ids / attention_mask
        padded_ids = []
        padded_mask = []

        for ids in encoded_list:
            pad_len = max_seq_len - len(ids)
            padded_ids.append([self.tokenizer.pad_token_id] * pad_len+ids)
            padded_mask.append([0] * pad_len+[1] * len(ids))

        batch = {
            "input_ids": torch.tensor(padded_ids, dtype=torch.long),
            "attention_mask": torch.tensor(padded_mask, dtype=torch.long)
        }

        return batch
    def run(self):
        responses, overflow_indices = self.process_generation(model_inputs)

        # 把生成后总长度超长的移到 finished
        self.manager.finish_many(overflow_indices)

    def process_generation(self, model_inputs, max_total_length=4096):
        """
        输入:
            model_inputs: build_model_inputs 生成的 batch（active 的对话）
            max_total_length: 输入+生成后的最大长度

        输出:
            responses: List[str] 生成的文本
            to_finish: List[int] 哪些对话超长，需要弹出
        """

        # === 1. 模型生成 ===
        generated = self.model.generate(
            **model_inputs,
            max_new_tokens=512
        )

        # === 2. 截取新生成内容 ===
        new_tokens_batch = []
        total_lengths = []
        to_finish = []  # 标记哪些轨迹要被弹出（会从 active 移到 finished）

        for i, (input_ids, output_ids) in enumerate(zip(model_inputs["input_ids"], generated)):
            input_len = len(input_ids)
            output_len = len(output_ids)
            total_len = output_len  # output_len == input_len + new_tokens_count

            total_lengths.append(total_len)

            # 若超长 → 标记弹出
            if total_len > max_total_length:
                to_finish.append(i)
                # 仍然截取出新生成部分，但不送入后续生成
                new_tokens = output_ids[input_len:]
            else:
                new_tokens = output_ids[input_len:]

            new_tokens_batch.append(new_tokens)

        # === 3. 解码 ===
        responses = self.tokenizer.batch_decode(
            new_tokens_batch,
            skip_special_tokens=True
        )

        return responses, to_finish


# class AgentState(TypedDict):
#     user_input:str


def think():
    user_input = env.get_current
    generated_ids = self.model.generate(
        **model_inputs,
        max_new_tokens=512
    )
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]
    responses = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
    return responses

def process_response(responses):
    return env.processresponses)

turns = 5
for i in range(turns):
    responses = think()
    result = process_response()

rea = reward(result)






