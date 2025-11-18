from tools import ALL_TOOLS
from tools.base import ToolExe
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM, BatchEncoding
from dataclasses import dataclass
from conversation_data import ConversationManager, ConversationItem
from environment import Environment

@dataclass
class GRPOConfig:
    model_name: str = "Qwen/Qwen2.5-0.5B-Instruct"
    tool_exe:ToolExe = None
    device: str = "cpu"
    num_generations: int = 1
    clip_eps: float = 0.2
    beta: float = 0.0         # GRPO KL 系数
    lr: float = 1e-5
    max_length: int = 512
    num_iterations: int = 1    # >1 时使用 old_action_log_probs
    use_ref_model: bool = True


class GRPOTrainer:
    def __init__(self, config: GRPOConfig):
        self.cfg = config

        # tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(config.model_name)

        # 主策略模型
        self.model = AutoModelForCausalLM.from_pretrained(config.model_name).to(config.device)

        # 参考模型（不会训练）
        if config.use_ref_model:
            self.ref_model = AutoModelForCausalLM.from_pretrained(config.model_name).to(config.device)
            self.ref_model.eval()
        else:
            self.ref_model = None

        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=config.lr)

        self.tool_exe = config.tool_exe
        self.cm = ConversationManager(tokenizer=self.tokenizer)
        # 环境
        self.env = Environment(self.tokenizer,tool_exe,self.cm)

        # 对话管理器


    def rollout_multi_step(self, max_length=128,trajectory=2, max_steps=2):
        """
        对 cm.active 中的所有对话进行多轮连续采样
        """
        conv = ConversationItem(
            system_prompt="你是一个数学推理助手。",
            tools="[]"
        )
        conv.add_message({"from": "human", "value": "你好，我有一个数学问题。1+1=？"})
        self.cm.active.add_conversation(conv)
        self.cm.active.duplicate(trajectory-1)
        for step in range(max_steps):
            print(f"\n=== 第 {step + 1} 轮采样 ===")

            # 构造 batch
            batch = self.cm.build_model_batch(max_length=max_length)
            if batch:
                batch=batch.to(self.cfg.device)
            else:
                print("break")
                break

            # 模型生成
            with torch.no_grad():
                generated_ids = self.model.generate(
                    **batch,
                    max_new_tokens=100,
                    top_p=0.9,
                    temperature=0.7
                )

            # 取新生成的部分
            gen_only = [
                output_ids[len(input_ids):]
                for input_ids, output_ids in zip(batch.input_ids, generated_ids)
            ]
            replies = self.tokenizer.batch_decode(gen_only, skip_special_tokens=True)

            # 写回每条轨迹
            self.env.process_responses(replies)
            # for i, conv in enumerate(self.cm.active.items):
            #
            #     conv.add_message({"from": "gpt", "value": replies[i]})

            print("本轮生成：", replies)


    # ---------------------------------------------------------------------
    # 计算 token log probability
    # ---------------------------------------------------------------------
    def get_action_log_probs(self, model, input_ids, attention_mask):
        out = model(input_ids, attention_mask=attention_mask)
        logits = out.logits  # (B, L, V)

        log_probs = F.log_softmax(logits, dim=-1)
        target = input_ids.unsqueeze(-1)
        token_log_probs = log_probs.gather(-1, target).squeeze(-1)

        return token_log_probs

    def rollout(self):
        """
        1) 用 rollout_multi_step() 采样轨迹
        2) 用 env.reward() 获取每条轨迹的 scalar reward
        3) build_grpo_batch() 得到 input_ids/attention/action_mask
        4) 构造 GRPO batch（优势，旧 log prob，ref log prob）
        """
        # === 1. 多轮生成（填充 self.cm.active 和 self.cm.finished）===
        self.rollout_multi_step()

        # === 2. group-level reward ===
        rewards = torch.tensor(self.env.reward(), dtype=torch.float32, device=self.cfg.device)

        # GRPO advantage：句子级别
        advantages = (rewards - rewards.mean()) / (rewards.std() + 1e-8)

        # === 3. build GRPO token batch with action_mask ===
        grpo_batch = self.cm.build_grpo_batch(max_length=self.cfg.max_length)
        grpo_batch = grpo_batch.to(self.cfg.device)

        ids = grpo_batch["input_ids"]
        mask = grpo_batch["attention_mask"]
        action_mask = grpo_batch["action_mask"]

        # === 4. old & ref action log probs（token级）===
        with torch.no_grad():
            old_lp = self.get_action_log_probs(self.model, ids, mask)
            if self.ref_model:
                ref_lp = self.get_action_log_probs(self.ref_model, ids, mask)
            else:
                ref_lp = None
        return BatchEncoding({
            "prompt_response_ids": ids,
            "attention_mask": mask,
            "action_mask": action_mask,
            "old_action_log_probs": old_lp,
            "ref_action_log_probs": ref_lp,
            "advantages": advantages,
        })

    # ---------------------------------------------------------------------
    # 计算 loss（PPO + GRPO KL 正则）
    # ---------------------------------------------------------------------
    def compute_loss(self, batch):
        ids = batch["prompt_response_ids"]
        mask = batch["attention_mask"]
        action_mask = batch["action_mask"]
        # 当前策略模型的 log π
        action_log_probs = self.get_action_log_probs(self.model, ids, mask)

        # KL 正则：log(ref) - log(policy)
        if self.cfg.beta != 0.0 and batch["ref_action_log_probs"] is not None:
            ref_lp = batch["ref_action_log_probs"]
            log_ratio = (ref_lp - action_log_probs) * action_mask
            k3 = (log_ratio.exp() - 1 - log_ratio) * action_mask  # GRPO KL
        else:
            k3 = 0

        # advantage
        advantages = batch["advantages"].unsqueeze(1)  # (B,1)

        # PPO old prob
        if self.cfg.num_iterations > 1:
            old_lp = batch["old_action_log_probs"]
        else:
            old_lp = action_log_probs.detach()

        coef_1 = torch.exp(action_log_probs - old_lp)
        coef_2 = torch.clamp(coef_1, 1 - self.cfg.clip_eps, 1 + self.cfg.clip_eps)
        per_token_loss1 = coef_1 * advantages.unsqueeze(1)  # 一个序列中每个token的优势是一样的
        per_token_loss2 = coef_2 * advantages.unsqueeze(1)
        per_token_loss = -torch.min(per_token_loss1, per_token_loss2)
        per_token_loss = per_token_loss * action_mask

        if self.cfg.beta != 0.0:
            per_token_loss = per_token_loss + self.cfg.beta * k3

        loss = per_token_loss.sum(dim=1) / action_mask.sum(dim=1)
        return loss.mean()

    # ---------------------------------------------------------------------
    # 单步训练 step
    # ---------------------------------------------------------------------
    def train_step(self):
        batch = self.rollout()
        loss = self.compute_loss(batch)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()

    # ---------------------------------------------------------------------
    # 高层 API：训练多个 step
    # ---------------------------------------------------------------------
    def train(self, steps=100):
        for i in range(steps):
            loss = self.train_step()
            print(f"[Step {i}] Loss = {loss:.6f}")


if __name__ == "__main__":
    tool_exe = ToolExe()
    tool_exe.register(IntroTool())
    tool_exe.register(ListUsersTool())

    cfg = GRPOConfig(
        model_name="Qwen/Qwen2.5-0.5B-Instruct",
        device="cpu",
        num_generations=1,
        lr=2e-5
    )

    trainer = GRPOTrainer(cfg)
    # trainer.rollout_multi_step()
    trainer.rollout()
    # 添加对话
    # conv = ConversationItem(system_prompt="你是数学助手。", tools="[]")
    # conv.add_message({"from": "human", "value": "请问 123+456 等于多少？"})
    # trainer.cm.active.add_conversation(conv)
    #
    # # 训练
    # trainer.train(steps=100)
