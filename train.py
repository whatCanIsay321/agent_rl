import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from conversation_data import ConversationItem, ConversationList, ConversationManager
from sampler import Sampler
from environment import Environment
from processor import process_function
from trainer_agent_grpo import AgentGRPOTrainer, GRPOAgentArguments


# ========== 示例奖励函数 ==========

def extract_answer(text):
    if "<answer>" in text:
        try:
            ans = text.split("<answer>")[-1].split("</answer>")[0].strip()
            return ans
        except:
            return ""
    return ""


def correctness_reward(prompts, responses, answers):
    """
    示例：如果模型输出的 <answer> 与答案一致 → 奖励 1，否则 0
    """
    results = []
    for p, r, a in zip(prompts, responses, answers):
        pred = extract_answer(r)
        results.append(1.0 if pred == a else 0.0)
    return results


# ========= 示例：digit reward ==========
def digit_reward(prompts, responses, answers):
    results = []
    for r in responses:
        pred = extract_answer(r)
        results.append(0.5 if pred.isdigit() else 0.0)
    return results


# ========== MAIN ==========

if __name__ == "__main__":

    # -----------------------------------------------------
    # 1. 加载模型与 tokenizer
    # -----------------------------------------------------
    model_name = "Qwen/Qwen2.5-0.5B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)

    # -----------------------------------------------------
    # 2. 构造 ConversationManager
    # -----------------------------------------------------
    conv = ConversationItem(
        system_prompt="你是一个专业 AI 助手。",
        tools="无",
        answer="42"     # 你也可以不加 answer
    )
    conv.add("human", "1+1等于几？请按格式输出")

    cl = ConversationList()
    cl.add_conversation(conv)

    cm = ConversationManager(active=cl, tokenizer=tokenizer)

    # -----------------------------------------------------
    # 3. Sampler / Environment（简单伪实现）
    # -----------------------------------------------------
    env = Environment(model, tokenizer, None, cm)
    sampler = Sampler(model, tokenizer, env, cm)

    # -----------------------------------------------------
    # 4. 定义 reward 函数
    # -----------------------------------------------------
    reward_funcs = [correctness_reward, digit_reward]

    # -----------------------------------------------------
    # 5. GRPO 配置
    # -----------------------------------------------------
    args = GRPOAgentArguments(
        device="cuda" if torch.cuda.is_available() else "cpu",
        sample_rounds=1,        # 每回合 rollout 轮数
        epoch=1,                # 训练轮次
        lr=1e-6,
        clip_eps=0.2,
        reward_weights=[1.0, 0.5],   # 多个 reward 函数的权重
    )

    # -----------------------------------------------------
    # 6. 创建 Trainer
    # -----------------------------------------------------
    trainer = AgentGRPOTrainer(
        model=model,
        tokenizer=tokenizer,
        cm=cm,
        sampler=sampler,
        reward_funcs=reward_funcs,
        args=args,
        target_tags=[("<answer>", "</answer>")]  # 只训练 <answer> 字段
    )

    # -----------------------------------------------------
    # 7. 开始训练！
    # -----------------------------------------------------
    trainer.train()

    print("训练结束！")
