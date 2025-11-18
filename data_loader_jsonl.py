# ============================================================
# data_loader_jsonl.py
# 多文件 JSONL → ConversationItem 的流式 DataLoader
# 支持无限迭代、shuffle、多卡训练(Accelerate)
# ============================================================

import json
import random
from itertools import cycle
import torch
from torch.utils.data import IterableDataset


# ============================================================
# jsonl 多文件流式生成器
# ============================================================

class JsonlMultiFileStream:
    """
    多文件 jsonl 流式加载，每次 yield 一行 json dict。
    - 支持 shuffle
    - 支持无限迭代
    - 不占内存（逐行读取）
    """
    def __init__(self, file_paths, shuffle=True):
        self.paths = list(file_paths)
        self.shuffle = shuffle

        if shuffle:
            random.shuffle(self.paths)

        # 无限循环的文件路径
        self.path_cycle = cycle(self.paths)

    def __iter__(self):
        return self._generator()

    def _generator(self):
        """无限循环读取 jsonl 文件"""
        for path in self.path_cycle:
            with open(path, "r", encoding="utf-8") as f:
                lines = f.readlines()

            if self.shuffle:
                random.shuffle(lines)

            for line in lines:
                line = line.strip()
                if not line:
                    continue
                yield json.loads(line)


# ============================================================
# JSONL → ConversationItem 的 IterableDataset
# ============================================================

class JsonlConversationDataset(IterableDataset):
    """
    适合大型 jsonl 数据，流式读取，不加载到内存。
    输入：JsonlMultiFileStream
    输出：ConversationItem
    """
    def __init__(self, stream, ConversationItemClass):
        super().__init__()
        self.stream = stream
        self.ConversationItemClass = ConversationItemClass

    def __iter__(self):
        for ex in self.stream:
            conv = self.ConversationItemClass(
                system_prompt=ex.get("system_prompt", ""),
                tools=ex.get("tools", "[]")
            )
            for m in ex.get("messages", []):
                conv.add_message(m)
            yield conv


# ============================================================
# collate_fn：直接返回列表（每卡 1 条）
# ============================================================

def conv_collate_fn(batch):
    """
    batch_size = world_size 时，
    batch 形如：
        [ConversationItem, ConversationItem, ...]
    每张 GPU 会取其中一条。
    """
    return batch


# ============================================================
# 构建可直接用于 Accelerate 的 DataLoader
# ============================================================

def build_jsonl_dataloader(jsonl_paths, ConversationItemClass,
                           batch_size=1, shuffle=True, num_workers=0):
    """
    返回：DataLoader
    """
    stream = JsonlMultiFileStream(
        file_paths=jsonl_paths,
        shuffle=shuffle
    )

    dataset = JsonlConversationDataset(
        stream=stream,
        ConversationItemClass=ConversationItemClass
    )

    from torch.utils.data import DataLoader

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,   # IterableDataset 禁止 shuffle
        num_workers=num_workers,
        collate_fn=conv_collate_fn
    )
