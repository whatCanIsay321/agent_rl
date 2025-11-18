from tools import ALL_TOOLS
from tools.base import ToolExe
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM, BatchEncoding
from dataclasses import dataclass
from conversation_data import ConversationManager, ConversationItem
from environment import Environment


# ============================================================
# 配置
# ============================================================
@dataclass
class GRPOConfig:
    model_name: str = "Qwen/Qwen2.5-0.5B-Instruct"
    tool_exe: ToolExe = None
    device: str = "cpu"

    # GRPO 超参
    clip_eps: float = 0.2
    beta: float = 0.0            # KL 系数
    lr: float = 1e-5

    # rollout_multi_step 多轮采样参数
    trajectory = 2
    max_steps = 2
    # rollout 采样参数

    num_iterations: int = 2 # >1 时 old_action_log_probs 生效

    # rollout 公用采样参数
    max_length: int = 128

    accumulation_steps: int = 4


    # 是否使用 ref model
    use_ref_model: bool = True


# ============================================================
# GRPO Trainer
# ============================================================
class GRPOTrainer:
    def __init__(self, config: GRPOConfig):
        self.cfg = config
        # ---------------- Tokenizer ---------------- #
        self.tokenizer = AutoTokenizer.from_pretrained(config.model_name)

        # ---------------- Policy Model ---------------- #
        self.model = AutoModelForCausalLM.from_pretrained(
            config.model_name
        ).to(config.device)

        # ---------------- Reference Model ---------------- #
        if config.use_ref_model:
            self.ref_model = AutoModelForCausalLM.from_pretrained(
                config.model_name
            ).to(config.device)
            self.ref_model.eval()
        else:
            self.ref_model = None

        # ---------------- Optimizer ---------------- #
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=config.lr)

        # ---------------- Tool System ---------------- #
        self.tool_exe = config.tool_exe or ToolExe()
        for ToolCls in ALL_TOOLS:
            self.tool_exe.register(ToolCls())

        # ---------------- Conversation Manager + Env ---------------- #
        self.cm = ConversationManager(tokenizer=self.tokenizer)
        self.env = Environment(self.tokenizer, self.tool_exe, self.cm)


    # ============================================================
    # 多轮 rollout（采样）
    # ============================================================
    def rollout_multi_step(self, conv):

        # 重置 active & finished
        self.cm.active.items = []
        self.cm.finished.items = []
        self.cm.active.add_conversation(conv)

        # 多轨迹复制
        if self.cfg.trajectory > 1:
            self.cm.active.duplicate(self.cfg.trajectory - 1)

        # 多轮 rollout
        for step in range(self.cfg.max_steps):
            print(f"\n=== 第 {step + 1} 轮采样 ===")

            batch = self.cm.build_model_batch(max_length=self.cfg.max_length)
            if not batch:
                print("No batch, break rollout")
                break

            batch = batch.to(self.cfg.device)

            # 模型生成
            with torch.no_grad():
                generated_ids = self.model.generate(
                    **batch,
                    max_new_tokens=100,
                    top_p=0.9,
                    temperature=0.7
                )

            # 提取增量 token
            gen_only = [
                output_ids[len(input_ids):]
                for input_ids, output_ids in zip(batch.input_ids, generated_ids)
            ]

            replies = self.tokenizer.batch_decode(gen_only, skip_special_tokens=True)

            # 写回对话轨迹
            self.env.process_responses(replies)
            print("本轮生成：", replies)


    # ============================================================
    # 修正确 token 对齐的 logprob（最重要）
    # ============================================================
    def get_action_log_probs(self, model, input_ids, attention_mask):
        output = model(input_ids, attention_mask=attention_mask)
        logits = output.logits
        log_probs = F.log_softmax(logits[:, :-1, :], dim=-1)
        log_probs_labels = log_probs.gather(dim=-1, index=input_ids[:, 1:].unsqueeze(-1))
        action_log_probs = log_probs_labels.squeeze(-1)
        return action_log_probs


    # ============================================================
    # rollout → 构造 GRPO batch
    # ============================================================
    def rollout(self, conv):
        # rollout 采样
        self.rollout_multi_step(conv)
        # 奖励（B, 1）
        rewards = torch.tensor(self.env.reward(), dtype=torch.float32, device=self.cfg.device)
        advantages = (rewards - rewards.mean()) / (rewards.std() + 1e-8)
        # 优势 (标准化)

        # 构造 batch（包含 action_mask）
        grpo_batch = self.cm.build_grpo_batch(max_length=self.cfg.max_length).to(self.cfg.device)

        ids = grpo_batch["input_ids"]
        mask = grpo_batch["attention_mask"]
        action_mask = grpo_batch["action_mask"]

        # 旧策略 logprob
        with torch.no_grad():
            old_lp = self.get_action_log_probs(self.model, ids, mask)
            ref_lp = (
                self.get_action_log_probs(self.ref_model, ids, mask)
                if self.ref_model else None
            )

        return BatchEncoding({
            "prompt_response_ids": ids,
            "attention_mask": mask,
            "action_mask": action_mask,
            "old_action_log_probs": old_lp,
            "ref_action_log_probs": ref_lp,
            "advantages": advantages,
        })


    # ============================================================
    # GRPO Loss
    # ============================================================
    def compute_loss(self, batch):
        ids = batch["prompt_response_ids"]
        mask = batch["attention_mask"]
        action_mask = batch["action_mask"][:, 1:]

        # 新策略 logprob
        action_log_probs = self.get_action_log_probs(self.model, ids, mask)
        advantages = batch["advantages"]          # (B, 1)
        advantages = advantages.unsqueeze(1)      # (B, 1, 1)? No, 保持 (B,1)

        # PPO clipped ratio
        old_lp = batch["old_action_log_probs"] if self.cfg.num_iterations > 1 else action_log_probs.detach()
        ratio = torch.exp(action_log_probs - old_lp)
        clipped_ratio = torch.clamp(ratio, 1 - self.cfg.clip_eps, 1 + self.cfg.clip_eps)

        per_token_loss = -torch.min(
            ratio * advantages,
            clipped_ratio * advantages
        ) * action_mask

        # GRPO KL 项
        if self.cfg.beta != 0.0 and batch["ref_action_log_probs"] is not None:
            ref_lp = batch["ref_action_log_probs"]
            log_ratio = (ref_lp - action_log_probs) * action_mask
            k3 = torch.exp(log_ratio) - 1 - log_ratio
            per_token_loss += self.cfg.beta * k3

        loss = per_token_loss.sum(dim=1) / action_mask.sum(dim=1)
        return loss.mean()


    # ============================================================
    # 单步训练
    # ============================================================
    # def train_step(self, conv):
    #     batch = self.rollout(conv)
    #     loss = self.compute_loss(batch)
    #
    #     self.optimizer.zero_grad()
    #     loss.backward()
    #     self.optimizer.step()
    #
    #     return loss.item()
    def train_step(self, conv):
        batch = self.rollout(conv)
        loss = self.compute_loss(batch)

        # 梯度累加：loss 需要归一化
        loss = loss / self.cfg.accumulation_steps

        loss.backward()
        return loss.item()

    # ============================================================
    # 训练循环
    # ============================================================
    # def train(self, conv, steps=100):
    #     for i in range(steps):
    #         loss = self.train_step(conv)
    #         print(f"[Step {i}] Loss = {loss:.6f}")
    def train(self, conv, steps=100):
        self.optimizer.zero_grad()

        for i in range(steps):
            loss = self.train_step(conv)

            # 满足梯度累加次数，更新模型
            if (i + 1) % self.cfg.accumulation_steps == 0:
                self.optimizer.step()
                self.optimizer.zero_grad()

            print(f"[Step {i}] Loss = {loss:.6f}")


# ============================================================
# 入口
# ============================================================
if __name__ == "__main__":
    conv = ConversationItem(
        system_prompt="你是一个数学助手。",
        tools="[]"
    )
    conv.add_message({"from": "human", "value": "1+1等于多少？"})

    tool_exe = ToolExe()
    for ToolCls in ALL_TOOLS:
        tool_exe.register(ToolCls())

    cfg = GRPOConfig(tool_exe=tool_exe)
    trainer = GRPOTrainer(cfg)

    # 测试 rollout
    # trainer.rollout_multi_step(
    #     conv=conv,
    #     trajectory=3,
    #     max_steps=2
    # )
    trainer.train(conv, steps=1000)
