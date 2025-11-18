from tools import ALL_TOOLS
from tools.base import ToolExe
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM, BatchEncoding
from dataclasses import dataclass
from torch.utils.data import Dataset, DataLoader
from conversation_data import ConversationManager, ConversationItem
from environment import Environment
from accelerate import Accelerator
from accelerate.utils import set_seed
from torch.utils.data import DataLoader# 你已有
from data_loader_jsonl import build_jsonl_dataloader
# ============================================================
# 配置
# ============================================================
@dataclass
class GRPOConfig:
    model_name: str = "Qwen/Qwen2.5-0.5B-Instruct"
    tool_exe: ToolExe = None
    device: str = "cpu"   # 会被 Accelerator 的 device 覆盖

    # GRPO 超参
    clip_eps: float = 0.2
    beta: float = 0.0            # KL 系数
    lr: float = 1e-5

    # rollout 多轮采样参数
    trajectory: int = 2          # 每个进程采样的轨迹数
    max_steps: int = 2           # 每条轨迹的最大交互轮数

    # rollout / 更新 参数
    num_iterations: int = 2      # 每次 rollout 后，在同一批数据上更新的次数
    max_length: int = 128        # prompt+response 最大长度

    # 梯度累加
    accumulation_steps: int = 4

    # 是否使用 ref model
    use_ref_model: bool = True

# ============================================================
# GRPO Trainer with Accelerate (每卡 + DataLoader 独立采样)
# ============================================================
class GRPOTrainer:
    def __init__(self, config: GRPOConfig):
        self.cfg = config

        # ---------------- Accelerator ---------------- #
        self.accelerator = Accelerator(
            gradient_accumulation_steps=self.cfg.accumulation_steps
        )
        # 用 accelerator 的 device 覆盖 config.device
        self.cfg.device = self.accelerator.device

        # 每个进程用不同的随机种子 → 采样结果不同
        base_seed = 42
        set_seed(base_seed + self.accelerator.process_index)

        # ---------------- Tokenizer ---------------- #
        self.tokenizer = AutoTokenizer.from_pretrained(config.model_name)

        # ---------------- Policy Model ---------------- #
        # 先在 CPU 上初始化，之后交给 accelerator.prepare
        model = AutoModelForCausalLM.from_pretrained(config.model_name)

        # ---------------- Optimizer ---------------- #
        optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr)

        # 让 accelerate 管理 model / optimizer（多卡 DDP & device）
        self.model, self.optimizer = self.accelerator.prepare(model, optimizer)

        # ---------------- Reference Model ---------------- #
        # ref model 不需要反向传播，用普通的 to(device) 就好
        if config.use_ref_model:
            self.ref_model = AutoModelForCausalLM.from_pretrained(
                config.model_name
            ).to(self.cfg.device)
            self.ref_model.eval()
            for p in self.ref_model.parameters():
                p.requires_grad_(False)
        else:
            self.ref_model = None

        # ---------------- Tool System ---------------- #
        if config.tool_exe is not None:
            self.tool_exe = config.tool_exe
        else:
            self.tool_exe = ToolExe()
            for ToolCls in ALL_TOOLS:
                self.tool_exe.register(ToolCls())

        # ---------------- Conversation Manager + Env ---------------- #
        # 每个进程各自一份，互不干扰
        self.cm = ConversationManager(tokenizer=self.tokenizer)
        self.env = Environment(self.tokenizer, self.tool_exe, self.cm)

    # ============================================================
    # 多轮 rollout（采样）—— 每个进程独立执行
    # ============================================================
    def rollout_multi_step(self, conv: ConversationItem):
        # 重置 active & finished
        self.cm.active.items = []
        self.cm.finished.items = []

        # ⚠️ 为安全起见，可以 clone 一份 conv（取决于你的 ConversationItem 是否会被 env 修改）
        # 这里假设 add_conversation 内部会复制，不直接改原对象
        self.cm.active.add_conversation(conv)

        # 多轨迹复制（本进程内部）
        if self.cfg.trajectory > 1:
            self.cm.active.duplicate(self.cfg.trajectory - 1)

        # 用 unwrap_model 拿到底层 HF 模型，用于 generate
        base_model = self.accelerator.unwrap_model(self.model)

        # 暂存 training 状态，用完恢复
        was_training = base_model.training
        base_model.eval()

        # 多轮 rollout
        for step in range(self.cfg.max_steps):
            batch = self.cm.build_model_batch(max_length=self.cfg.max_length)
            if batch is None or len(batch["input_ids"]) == 0:
                if self.accelerator.is_main_process:
                    print(f"[Rank {self.accelerator.process_index}] No batch, break rollout")
                break

            batch = batch.to(self.cfg.device)

            # 模型生成（无梯度）
            with torch.no_grad():
                generated_ids = base_model.generate(
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
            if self.accelerator.is_main_process:
                print(f"[Rank {self.accelerator.process_index}] 第 {step+1} 轮生成: {replies}")

        # 恢复训练模式
        base_model.train(was_training)

    # ============================================================
    # log prob 计算
    # ============================================================
    def get_action_log_probs(self, model, input_ids, attention_mask):
        output = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = output.logits                          # (B, T, V)

        log_probs = F.log_softmax(logits[:, :-1, :], dim=-1)  # (B, T-1, V)

        log_probs_labels = log_probs.gather(
            dim=-1, index=input_ids[:, 1:].unsqueeze(-1)
        )                                               # (B, T-1, 1)
        action_log_probs = log_probs_labels.squeeze(-1) # (B, T-1)
        return action_log_probs

    # ============================================================
    # rollout → 构造 GRPO batch（每个进程独立）
    # ============================================================
    def rollout(self, conv: ConversationItem) -> BatchEncoding:
        # 1. 用当前策略 rollout（本 rank 独立）
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
            advantages = rewards - rewards.mean()

        # 4. 构造 batch（包含 action_mask）
        grpo_batch = self.cm.build_grpo_batch(
            max_length=self.cfg.max_length
        ).to(self.cfg.device)

        ids = grpo_batch["input_ids"]          # (B, T)
        mask = grpo_batch["attention_mask"]    # (B, T)
        action_mask = grpo_batch["action_mask"]  # (B, T)

        # 5. 旧策略 logprob（固定为 rollout 时的策略）
        base_model = self.accelerator.unwrap_model(self.model)
        with torch.no_grad():
            old_lp = self.get_action_log_probs(base_model, ids, mask)  # (B, T-1)

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

        # 新策略 logprob（用 self.model，已经被 prepare 包装，会自动多卡）
        action_log_probs = self.get_action_log_probs(self.model, ids, mask)  # (B, T-1)

        # Advantage (B,) -> (B,1)，方便广播
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
            log_ratio = (ref_lp - action_log_probs) * action_mask
            k3 = torch.exp(log_ratio) - 1 - log_ratio
            per_token_loss += self.cfg.beta * k3

        per_sample_denominator = action_mask.sum(dim=1).clamp(min=1)  # 防止除零
        loss = per_token_loss.sum(dim=1) / per_sample_denominator     # (B,)
        return loss.mean()

    # ============================================================
    # 单次 rollout 上的多次更新（所有 rank 各自用自己的 batch）
    # ============================================================
    def train_step(self, batch: BatchEncoding, rollout_idx: int):
        avg_loss = 0.0

        for it in range(self.cfg.num_iterations):
            # accumulate 里自动处理梯度累加 + 多卡同步
            with self.accelerator.accumulate(self.model):
                loss = self.compute_loss(batch)

                self.accelerator.backward(loss)

                if self.accelerator.sync_gradients:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    self.optimizer.step()
                    self.optimizer.zero_grad()

            avg_loss += loss.item()

            if self.accelerator.is_main_process:
                print(f"[Rollout {rollout_idx} | Iter {it}] Loss = {loss.item():.6f}")

        avg_loss /= max(self.cfg.num_iterations, 1)
        return avg_loss

    # ============================================================
    # 训练循环：DataLoader + 每张卡独立 rollout
    # ============================================================
    def train(self, dataloader: DataLoader, epochs: int = 1):
        """
        dataloader: 未经过 accelerator.prepare 的 DataLoader
        每个 rank 会看到自己分片后的数据，因此每张卡的 ConversationItem 不同。
        """
        # 让 accelerate 处理 DataLoader 的分片
        dataloader = self.accelerator.prepare(dataloader)

        for epoch in range(epochs):
            if self.accelerator.is_main_process:
                print(f"\n===== Epoch {epoch} =====")

            for step, batch in enumerate(dataloader):
                # batch 是 List[ConversationItem]，长度=batch_size（这里推荐=1）
                for conv in batch:
                    local_batch = self.rollout(conv)
                    avg_loss = self.train_step(local_batch, rollout_idx=step)

                if self.accelerator.is_main_process:
                    print(f"[Epoch {epoch} | Step {step}] Avg Loss (rank0 view) = {avg_loss:.6f}")



from conversation_data import ConversationItem

if __name__ == "__main__":
    dataloader = build_jsonl_dataloader(
        jsonl_paths=["data/train1.jsonl", "data/train2.jsonl"],
        ConversationItemClass=ConversationItem,
        batch_size=1,  # 每个 rank 每 step 1 条样本
        shuffle=True
    )

    cfg = GRPOConfig()
    trainer = GRPOTrainer(cfg)

    # 交给 trainer 内部的 self.accelerator 处理多卡 & 梯度累加
    trainer.train(dataloader, epochs=1)
