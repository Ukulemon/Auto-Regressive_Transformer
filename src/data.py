"""Dataset and preprocessing utilities for the Transformer language model."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import torch
from torch.utils.data import DataLoader, Dataset, random_split

from .tokenizer import BaseTokenizer, build_tokenizer, pad_sequences


@dataclass
class DatasetConfig:
    data_path: str
    tokenizer: str = "word"
    sequence_length: int = 128
    lowercase: bool = True
    vocab_size: int = 2000
    min_frequency: int = 2
    remove_stopwords: bool = True


class TextSequenceDataset(Dataset):
    """PyTorch dataset producing (input, target) token id sequences."""

    def __init__(self, token_ids: List[int], sequence_length: int, pad_token_id: int) -> None:
        """构造函数：根据整段语料的编码序列切分出模型训练所需的输入与标签。

        参数说明：
        - ``token_ids``：来自``preprocess_corpus``函数返回的整段文本标记ID列表。
        - ``sequence_length``：由配置或命令行传入的序列长度，用于控制每个样本的长度。
        - ``pad_token_id``：由分词器提供的填充标记ID，用于后续批处理填充。
        """
        print(f"Number of tokens: {len(token_ids)}")
        if sequence_length < 2:
            raise ValueError("Sequence length must be >= 2")
        self.token_ids = token_ids
        self.sequence_length = sequence_length
        self.pad_token_id = pad_token_id

        self.inputs: List[List[int]] = []
        self.targets: List[List[int]] = []
        max_start = max(0, len(self.token_ids) - sequence_length)
        for start in range(0, max_start):
            chunk = token_ids[start : start + sequence_length + 1]
            if len(chunk) < sequence_length + 1:
                continue
            self.inputs.append(chunk[:-1])
            self.targets.append(chunk[1:])

    def __len__(self) -> int:  # pragma: no cover - simple delegation
        """返回数据集中样本数量，对应于切分后的序列对数量。"""
        return len(self.inputs)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """根据索引返回单个样本的输入与目标张量，用于``DataLoader``迭代。

        参数说明：
        - ``idx``：由``DataLoader``自动生成的样本索引。
        返回值为模型的输入序列张量以及对应的下一步预测目标张量。
        """
        src = torch.tensor(self.inputs[idx], dtype=torch.long)
        tgt = torch.tensor(self.targets[idx], dtype=torch.long)
        return src, tgt


def collate_batch(batch, pad_token_id: int):
    """自定义批处理函数：将若干样本对齐并堆叠成批次张量。

    参数说明：
    - ``batch``：由``DataLoader``采样得到的样本列表，内部元素来自``TextSequenceDataset.__getitem__``。
    - ``pad_token_id``：来自分词器的填充标记ID，用于对齐不同长度的序列。
    返回处理后的输入与目标批次张量。
    """
    inputs, targets = zip(*batch)
    input_padded = pad_sequences([seq.tolist() for seq in inputs], pad_token_id)
    target_padded = pad_sequences([seq.tolist() for seq in targets], pad_token_id)
    return input_padded, target_padded


def read_corpus(data_path: str) -> str:
    """读取文本语料文件并返回完整字符串内容。

    参数说明：
    - ``data_path``：来自配置或命令行参数指定的语料文件路径。
    """
    path = Path(data_path)
    if not path.exists():
        raise FileNotFoundError(f"Data file not found: {data_path}")
    return path.read_text(encoding="utf-8")


def preprocess_corpus(
    text: str,
    tokenizer_type: str,
    lowercase: bool = True,
    vocab_size: int = 2000,
    min_frequency: int = 2,
) -> Tuple[BaseTokenizer, List[int]]:
    """根据用户选择的分词器类型对语料做预处理并转换为ID序列。

    参数说明：
    - ``text``：来自``read_corpus``读取的原始文本。
    - ``tokenizer_type``：来源于命令行参数或配置，决定使用字符、词或子词分词。
    - ``lowercase``/``vocab_size``/``min_frequency``：配置中设定的额外分词器构建参数。
    返回训练时使用的分词器实例以及包含首尾特殊符号的整段ID序列。
    """
    tokenizer_kwargs = {
        "lowercase": lowercase,
    }
    if tokenizer_type in {"bpe", "subword"}:
        tokenizer_kwargs.update({"vocab_size": vocab_size, "min_frequency": min_frequency})
    tokenizer = build_tokenizer(tokenizer_type, **tokenizer_kwargs)
    tokenizer.fit([text])
    encoded = tokenizer.encode(text, add_special_tokens=False)
    encoded = [tokenizer.bos_token_id] + encoded + [tokenizer.eos_token_id]
    return tokenizer, encoded


def train_val_split(tokens: List[int], split_ratio: float = 0.9) -> Tuple[List[int], List[int]]:
    """按照比例将编码后的整段序列切分为训练与验证部分。
    参数说明：
    - ``tokens``：由``preprocess_corpus``生成的整段ID列表。
    - ``split_ratio``：来自配置的训练集占比，默认90%。
    返回训练与验证两部分的ID列表。
    """
    split_index = int(len(tokens) * split_ratio)
    train_tokens = tokens[:split_index]
    val_tokens = tokens[split_index:]
    print(f"Number of training tokens: {len(train_tokens)}")
    print(f"Number of validation tokens: {len(val_tokens)}")
    return train_tokens, val_tokens


def create_dataloaders(
    token_ids: List[int],
    tokenizer: BaseTokenizer,
    batch_size: int,
    sequence_length: int,
    num_workers: int = 0,
    split_ratio: float = 0.9,
) -> Tuple[DataLoader, DataLoader]:
    """创建训练与验证数据加载器，封装批处理、打乱等逻辑。

    参数说明：
    - ``token_ids``：来自``preprocess_corpus``的整段标记ID列表。
    - ``tokenizer``：在预处理阶段构建好的分词器，用于提供填充符号。
    - ``batch_size``/``sequence_length``/``num_workers``/``split_ratio``：用户配置或命令行提供的训练参数。
    返回可直接供训练循环使用的训练与验证``DataLoader``。
    """
    full_dataset = TextSequenceDataset(token_ids, sequence_length, tokenizer.pad_token_id)
    if len(full_dataset) < 2:
        raise ValueError(
            "Not enough sequences to create both training and validation splits. "
            "Provide more data or reduce sequence length."
        )

    train_size = int(len(full_dataset) * split_ratio)
    val_size = len(full_dataset) - train_size
    if val_size == 0:
        val_size = 1
        train_size -= 1
    if train_size <= 0:
        raise ValueError(
            "Training split would be empty. Adjust split ratio or increase dataset size."
        )

    train_dataset, val_dataset = random_split(
        full_dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42),
    )

    def collate_fn(batch):
        """内部函数：包装``collate_batch``并固定填充标记ID，供``DataLoader``回调。"""
        return collate_batch(batch, tokenizer.pad_token_id)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=num_workers,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=num_workers,
        drop_last=False,
    )
    return train_loader, val_loader
