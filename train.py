import os
from accelerate import Accelerator
from grpo_trainer import GRPOTrainer, GRPOConfig
from conversation_data import ConversationItem


# ============================================================
# 1) 这里定义每张卡不同的输入（关键）
# ============================================================
def get_inputs_for_each_gpu():
    """
    每个 GPU 独立 rollout 的输入。
    如果 GPU 数量大于此列表，输入会自动循环分配。
    """
    return [
        "1+1 等于多少？",
        "什么是勾股定理？",
        "积分 ∫x dx 等于多少？",
        "说明泰勒展开的意义。",
        "什么是随机梯度下降？",
        "圆的面积公式为什么是 πr^2？",
        "证明三角形内角和是 180°。",
        "极限 lim(x->0) sin(x)/x 等于多少？"
    ]


# ============================================================
# 2) 主训练函数
# ============================================================
def main():

    # accelerator 负责分布式同步、混合精度等
    accelerator = Accelerator()

    # ---------------------------
    # 让每个 GPU 拿不同输入
    # ---------------------------
    all_inputs = get_inputs_for_each_gpu()
    local_input = all_inputs[accelerator.process_index % len(all_inputs)]

    # 创建对话对象 (每 GPU 独立)
    conv = ConversationItem(
        system_prompt="你是一个数学助手。",
        tools="[]"
    )
    conv.add_message({"from": "human", "value": local_input})

    if accelerator.is_main_process:
        print(f"Total GPUs: {accelerator.num_processes}")
        print(f"This GPU({accelerator.process_index}) gets input: {local_input}")

    # ============================================================
    # 3) initialize trainer (model not yet on devices)
    # ============================================================
    cfg = GRPOConfig()
    trainer = GRPOTrainer(cfg)

    # ============================================================
    # 4) accelerate 接管模型 + optimizer
    # ============================================================
    trainer.model, trainer.optimizer = accelerator.prepare(
        trainer.model,
        trainer.optimizer
    )

    # ref model 不需要梯度同步，但要移到设备
    if trainer.ref_model is not None:
        trainer.ref_model = trainer.ref_model.to(accelerator.device)

    # ============================================================
    # 5) 主训练循环
    # ============================================================
    steps = 1000
    accumulation = cfg.accumulation_steps

    trainer.optimizer.zero_grad()

    for step in range(steps):

        # ------- rollout（多卡独立，不同步） -------
        rollout_batch = trainer.rollout(conv)

        # ------- 计算 loss（会在所有 GPU all-reduce） -------
        loss = trainer.compute_loss(rollout_batch)
        loss = loss / accumulation

        # ------- backward -------
        accelerator.backward(loss)

        # ------- optimizer step with grad accumulation -------
        if (step + 1) % accumulation == 0:
            trainer.optimizer.step()
            trainer.optimizer.zero_grad()

        # ------- logging (main GPU only) -------
        if accelerator.is_main_process:
            print(f"[Step {step}] Loss = {loss.item():.6f}")

    # ============================================================
    # 6) 保存最终模型（只在主进程执行）
    # ============================================================
    if accelerator.is_main_process:
        save_dir = "grpo_checkpoint_final"
        accelerator.save_state(save_dir)
        trainer.model.save_pretrained(os.path.join(save_dir, "policy"))
        print(f"Model saved at {save_dir}")


if __name__ == "__main__":
    main()
