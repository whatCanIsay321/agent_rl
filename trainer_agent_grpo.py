import torch
import torch.nn.functional as F
from dataclasses import dataclass
from typing import List, Any
from processor import process_function
from torch.optim import Adam


@dataclass
class GRPOAgentArguments:
    device: str = "cuda"
    lr: float = 1e-6
    epoch: int = 1
    sample_rounds: int = 2      # 每次 rollout 的轮数
    max_length: int = 2048
    clip_eps: float = 0.2
    beta: float = 0.0           # KL penalty
    reward_weights: List[float] = None


class AgentGRPOTrainer:
    """
    全新版本：适配 ConversationManager + Sampler，用于训练 agent 生成中指定字段（如 <answer> 标签）
    """

    def __init__(self,
                 model,
                 tokenizer,
                 cm,              # ConversationManager
                 sampler,         # Sampler
                 reward_funcs,
                 args: GRPOAgentArguments,
                 target_tags      # 想要训练的字段，如 [("<answer>", "</answer>")]
                 ):

        self.model = model.to(args.device)
        self.tokenizer = tokenizer
        self.cm = cm
        self.sampler = sampler
        self.reward_funcs = reward_funcs
        self.args = args
        self.target_tags = target_tags

        self.optimizer = Adam(self.model.parameters(), lr=self.args.lr)

        # KL reference model
        self.ref_model = None
        if args.beta != 0:
            from copy import deepcopy
            self.ref_model = deepcopy(model).eval()

    # ============================================================
    # 1. 从 ConversationManager 抽取最终 assistant 输出
    # ============================================================
    def extract_final_responses(self):
        responses = []
        prompts = []

        for conv in self.cm.active.items + self.cm.finished.items:
            # 获取完整对话的 prompt 部分
            encoded = self.cm.process_single(conv.export(), self.args.max_length)
            prompts.append(encoded)

            # 获取最后 assistant 输出字段
            last_msg = ""
            for m in reversed(conv.conversations):
                if m.from_ == "gpt":
                    last_msg = m.value
                    break
            responses.append(last_msg)

        return prompts, responses

    # ============================================================
    # 2. 计算奖励（多个 reward_func）
    # ============================================================
    def compute_rewards(self, prompts, responses):
        rewards_per_func = []

        for reward_func in self.reward_funcs:
            rewards = reward_func(prompts, responses, answers=[""] * len(prompts))
            rewards_per_func.append(rewards)

        rewards = torch.tensor(rewards_per_func).sum(0).to(self.args.device)
        return rewards

    # ============================================================
    # 3. 构建训练样本（tokenization + label mask）
    # ============================================================
    def build_training_batch(self, conversation):

        processed = process_function(
            conversation=conversation,
            tokenizer=self.tokenizer,
            target_tags=self.target_tags,
            max_length=self.args.max_length
        )

        input_ids = torch.tensor(processed["input_ids"], dtype=torch.long).to(self.args.device)
        labels = torch.tensor(processed["labels"], dtype=torch.long).to(self.args.device)
        mask = torch.tensor(processed["attention_mask"], dtype=torch.long).to(self.args.device)

        return input_ids.unsqueeze(0), labels.unsqueeze(0), mask.unsqueeze(0)

    # ============================================================
    # 4. Token log-prob
    # ============================================================
    def get_log_probs(self, model, input_ids, attention_mask):
        outputs = model(input_ids, attention_mask=attention_mask)
        logits = outputs.logits[:, :-1]
        log_probs = F.log_softmax(logits, dim=-1)
        return log_probs.gather(-1, input_ids[:, 1:].unsqueeze(-1)).squeeze(-1)

    # ============================================================
    # 5. 单步优化
    # ============================================================
    def update_policy(self, logp, old_logp, advantages, mask):
        ratio = torch.exp(logp - old_logp)
        clipped_ratio = torch.clamp(ratio, 1 - self.args.clip_eps, 1 + self.args.clip_eps)

        loss1 = -ratio * advantages.unsqueeze(1)
        loss2 = -clipped_ratio * advantages.unsqueeze(1)
        loss = torch.min(loss1, loss2)

        loss = (loss * mask).sum() / mask.sum()

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()

    # ============================================================
    # 6. 主训练循环
    # ============================================================
    def train(self):

        for epoch in range(self.args.epoch):

            # -----------------------
            # (1) 多轮采样 rollout
            # -----------------------
            self.sampler.sample(rounds=self.args.sample_rounds)

            # -----------------------
            # (2) 抽取最终输出
            # -----------------------
            convs = self.cm.active.items + self.cm.finished.items
            prompts, responses = self.extract_final_responses()

            # -----------------------
            # (3) 奖励
            # -----------------------
            rewards = self.compute_rewards(prompts, responses)
            advantages = (rewards - rewards.mean()) / (rewards.std() + 1e-8)

            # -----------------------
            # (4) 逐条对话构建训练数据
            # -----------------------
            for conv, adv in zip(convs, advantages):

                input_ids, labels, mask = self.build_training_batch(conv.export())

                # old policy logp
                with torch.no_grad():
                    old_logp = self.get_log_probs(self.model, input_ids, mask)

                # new policy logp
                logp = self.get_log_probs(self.model, input_ids, mask)

                loss = self.update_policy(logp, old_logp, adv, (labels != -100).long())

                print(f"[Epoch {epoch}] Loss={loss:.4f}  Reward={adv.item():.4f}")

        print("Training completed.")
