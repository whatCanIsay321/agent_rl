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
    trajectory: int = 2          # 采样的轨迹数
    max_steps: int = 2           # 每条轨迹的最大交互轮数

    # rollout 采样 / 更新 参数
    num_iterations: int = 2      # 每次 rollout 后，在同一批数据上更新的次数
    max_length: int = 128        # prompt+response 最大长度

    # 梯度累加
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
            for p in self.ref_model.parameters():
                p.requires_grad_(False)
        else:
            self.ref_model = None

        # ---------------- Optimizer ---------------- #
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=config.lr)

        # ---------------- Tool System ---------------- #
        # 如果外部已经传入一个 ToolExe，就认为已经注册好工具，不再重复注册
        if config.tool_exe is not None:
            self.tool_exe = config.tool_exe
        else:
            self.tool_exe = ToolExe()
            for ToolCls in ALL_TOOLS:
                self.tool_exe.register(ToolCls())

        # ---------------- Conversation Manager + Env ---------------- #
        self.cm = ConversationManager(tokenizer=self.tokenizer)
        self.env = Environment(self.tokenizer, self.tool_exe, self.cm)

    # ============================================================
    # 多轮 rollout（采样）
    # ============================================================
    def rollout_multi_step(self, conv: ConversationItem):
        # 重置 active & finished
        self.cm.active.items = []
        self.cm.finished.items = []
        self.cm.active.add_conversation(conv)

        # 多轨迹复制
        if self.cfg.trajectory > 1:
            self.cm.active.duplicate(self.cfg.trajectory - 1)

        # rollout 前暂存 training 状态，用完恢复
        was_training = self.model.training
        self.model.eval()

        # 多轮 rollout
        for step in range(self.cfg.max_steps):
            print(f"\n=== 第 {step + 1} 轮采样 ===")

            batch = self.cm.build_model_batch(max_length=self.cfg.max_length)
            if batch is None or len(batch["input_ids"]) == 0:
                print("No batch, break rollout")
                break

            batch = batch.to(self.cfg.device)

            # 模型生成（无梯度）
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

        # 恢复训练模式
        self.model.train(was_training)

    # ============================================================
    # 修正确 token 对齐的 logprob（最重要）
    # ============================================================
    def get_action_log_probs(self, model, input_ids, attention_mask):
        """
        input_ids:      (B, T)
        attention_mask: (B, T)
        返回:
        action_log_probs: (B, T-1) 对应每个 token (从第1个到最后一个) 的 log p(token_t | <t)
        """
        output = model(input_ids, attention_mask=attention_mask)
        logits = output.logits                          # (B, T, V)

        # 预测 token_t 的 logits 位于位置 t-1
        log_probs = F.log_softmax(logits[:, :-1, :], dim=-1)  # (B, T-1, V)

        # input_ids[:, 1:] 是要预测的 token 序列
        log_probs_labels = log_probs.gather(
            dim=-1, index=input_ids[:, 1:].unsqueeze(-1)
        )                                               # (B, T-1, 1)
        action_log_probs = log_probs_labels.squeeze(-1) # (B, T-1)
        return action_log_probs

    # ============================================================
    # rollout → 构造 GRPO batch
    # ============================================================
    def rollout(self, conv: ConversationItem):
        # 1. 用当前策略 rollout
        self.rollout_multi_step(conv)

        # 2. 奖励（假设 env.reward() 返回一维列表或张量，长度为 B）
        rewards = torch.tensor(
            self.env.reward(),
            dtype=torch.float32,
            device=self.cfg.device
        )

        # 3. Advantage 计算：注意 batch_size=1 的情况
        if rewards.numel() > 1:
            advantages = (rewards - rewards.mean()) / (rewards.std() + 1e-8)
        else:
            # 单样本时直接使用 centered reward
            advantages = rewards - rewards.mean()

        # 4. 构造 batch（包含 action_mask）
        grpo_batch = self.cm.build_grpo_batch(
            max_length=self.cfg.max_length
        ).to(self.cfg.device)

        ids = grpo_batch["input_ids"]          # (B, T)
        mask = grpo_batch["attention_mask"]    # (B, T)
        action_mask = grpo_batch["action_mask"]  # (B, T)

        # 5. 旧策略 logprob（固定为 rollout 时的策略）
        with torch.no_grad():
            old_lp = self.get_action_log_probs(self.model, ids, mask)  # (B, T-1)

            if self.ref_model is not None and self.cfg.beta != 0.0:
                ref_lp = self.get_action_log_probs(self.ref_model, ids, mask)
            else:
                ref_lp = None

        return BatchEncoding({
            "prompt_response_ids": ids,
            "attention_mask": mask,
            "action_mask": action_mask,
            "old_action_log_probs": old_lp,
            "ref_action_log_probs": ref_lp,
            "advantages": advantages,  # (B,)
        })

    # ============================================================
    # GRPO Loss
    # ============================================================
    def compute_loss(self, batch: BatchEncoding):
        ids = batch["prompt_response_ids"]      # (B, T)
        mask = batch["attention_mask"]          # (B, T)

        # action_mask 对齐到 action_log_probs 的长度 (T-1)
        action_mask = batch["action_mask"][:, 1:]   # (B, T-1)

        # 新策略 logprob
        action_log_probs = self.get_action_log_probs(self.model, ids, mask)  # (B, T-1)

        # Advantage shape 调整，(B,) -> (B,1)，方便广播到 (B, T-1)
        advantages = batch["advantages"]            # (B,)
        advantages = advantages.unsqueeze(1)        # (B,1)

        # old_policy：如果 batch 里有缓存，就用缓存；否则退化成 on-policy
        if "old_action_log_probs" in batch and batch["old_action_log_probs"] is not None:
            old_lp = batch["old_action_log_probs"]      # (B, T-1)
        else:
            old_lp = action_log_probs.detach()

        ratio = torch.exp(action_log_probs - old_lp)              # (B, T-1)
        clipped_ratio = torch.clamp(ratio, 1 - self.cfg.clip_eps, 1 + self.cfg.clip_eps)

        # per-token loss，乘上 action_mask 只在有效 token 上起作用
        per_token_loss = -torch.min(
            ratio * advantages,
            clipped_ratio * advantages
        ) * action_mask  # (B, T-1)

        # GRPO KL 项（与 ref_model 的 KL 正则）
        if self.cfg.beta != 0.0 and batch.get("ref_action_log_probs", None) is not None:
            ref_lp = batch["ref_action_log_probs"]  # (B, T-1)
            # log_ratio = log(π_ref / π) = ref_lp - action_log_probs
            log_ratio = (ref_lp - action_log_probs) * action_mask
            # k3 = exp(log_ratio) - 1 - log_ratio
            k3 = torch.exp(log_ratio) - 1 - log_ratio
            per_token_loss += self.cfg.beta * k3

        # 对每个样本做 token 级别平均
        per_sample_denominator = action_mask.sum(dim=1).clamp(min=1)  # 防止除零
        loss = per_token_loss.sum(dim=1) / per_sample_denominator     # (B,)
        return loss.mean()

    # ============================================================
    # 单步训练：在同一个 batch 上多次更新
    # ============================================================
    def train_step(self, batch: BatchEncoding):
        """
        在同一个 batch 上做 self.cfg.num_iterations 次更新。
        注意：这里不再做 rollout，rollout 在 train() 里统一调用。
        """
        total_loss = 0.0

        for it in range(self.cfg.num_iterations):
            loss = self.compute_loss(batch)  # 标量

            # 梯度累加
            loss_scaled = loss / self.cfg.accumulation_steps
            loss_scaled.backward()

            total_loss += loss.item()

        # 返回平均 loss，方便打印
        return total_loss / self.cfg.num_iterations

    # ============================================================
    # 训练循环：一次 rollout，多次更新
    # ============================================================
    def train(self, conv: ConversationItem, steps: int = 100):
        """
        steps: 要做多少次 rollout
        每次 rollout 之后，在同一批数据上做 cfg.num_iterations 次更新
        """
        self.optimizer.zero_grad()
        update_step = 0  # 记录总的 “backward 次数”（用于梯度累加）

        for i in range(steps):
            # 1. 用当前策略 rollout 一次，得到一个 batch（含 old_action_log_probs）
            batch = self.rollout(conv)

            # 2. 在这个 batch 上做多次更新
            avg_loss = self.train_step(batch)
            update_step += self.cfg.num_iterations

            # 3. 按照 accumulation_steps 做一次参数更新
            if update_step % self.cfg.accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()
                self.optimizer.zero_grad()

            print(f"[Rollout {i}] Avg Loss = {avg_loss:.6f}")

        # 处理最后一批不足 accumulation_steps 的残余梯度
        if update_step % self.cfg.accumulation_steps != 0:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            self.optimizer.zero_grad()


# ============================================================
# 入口
# ============================================================
if __name__ == "__main__":
    conv = ConversationItem(
        system_prompt="你是一个数学助手。",
        tools="[]"
    )
    conv.add_message({"from": "human", "value": "1+1等于多少？"})

    # 让 GRPOTrainer 自己创建并注册工具，避免重复注册
    cfg = GRPOConfig(tool_exe=None, device="cpu")
    trainer = GRPOTrainer(cfg)

    # 训练：steps 是 rollout 次数，每次 rollout 上做 cfg.num_iterations 次更新
    trainer.train(conv, steps=1000)
