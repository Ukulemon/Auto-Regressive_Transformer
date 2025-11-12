"""Utility helpers for training and inference."""
from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, Optional

import torch


@dataclass
class TrainingConfig:
    data_path: str
    tokenizer: str = "word"
    sequence_length: int = 128
    batch_size: int = 16
    d_model: int = 512
    nhead: int = 8
    num_layers: int = 6
    dim_feedforward: int = 2048
    dropout: float = 0.1
    max_seq_len: int = 512
    learning_rate: float = 3e-4
    weight_decay: float = 0.01
    epochs: int = 10
    gradient_accumulation_steps: int = 1
    max_grad_norm: float = 1.0
    patience: int = 3
    warmup_steps: int = 0
    cosine_t_max: int = 10
    min_lr: float = 1e-5
    num_workers: int = 0
    gradient_checkpointing: bool = False
    resume_from: Optional[str] = None
    pretrained_weights: Optional[str] = None
    output_dir: str = "checkpoints"
    max_generate_length: int = 200
    temperature: float = 1.0
    top_k: Optional[int] = None
    mixed_precision: bool = True

    def save(self, path: str | Path) -> None:
        """将训练配置保存为JSON文件，方便后续复现。

        参数说明：
        - ``path``：来自用户指定的输出路径。
        """
        Path(path).write_text(json.dumps(asdict(self), indent=2), encoding="utf-8")

    @staticmethod
    def load(path: str | Path) -> "TrainingConfig":
        """从JSON文件加载训练配置。

        参数说明：
        - ``path``：指向先前保存配置文件的路径。
        """
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return TrainingConfig(**data)


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler],
    scaler: Optional[torch.cuda.amp.GradScaler],
    epoch: int,
    best_val_loss: float,
    config: TrainingConfig,
    path: str | Path,
) -> None:
    """保存训练检查点，包含模型、优化器、调度器和缩放器状态。

    参数说明：
    - ``model``/``optimizer``/``scheduler``/``scaler``：来自训练循环中的核心组件。
    - ``epoch``/``best_val_loss``：当前epoch编号与最佳验证损失，供恢复训练时参考。
    - ``config``：训练配置数据类实例。
    - ``path``：用户配置的检查点存储路径。
    """
    checkpoint = {
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict() if scheduler else None,
        "scaler_state": scaler.state_dict() if scaler else None,
        "epoch": epoch,
        "best_val_loss": best_val_loss,
        "config": asdict(config),
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, path)


def load_checkpoint(
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer],
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler],
    scaler: Optional[torch.cuda.amp.GradScaler],
    path: str | Path,
    device: torch.device,
) -> Dict[str, Any]:
    """从检查点文件中恢复模型与优化状态。

    参数说明：
    - ``model``/``optimizer``/``scheduler``/``scaler``：训练循环中的组件实例，将被写入检查点中的参数。
    - ``path``：检查点文件路径，由用户指定或自动保存。
    - ``device``：当前训练设备，用于正确加载权重。
    返回检查点完整字典，供调用方读取额外信息。
    """
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint["model_state"])
    if optimizer and checkpoint.get("optimizer_state"):
        optimizer.load_state_dict(checkpoint["optimizer_state"])
    if scheduler and checkpoint.get("scheduler_state"):
        scheduler.load_state_dict(checkpoint["scheduler_state"])
    if scaler and checkpoint.get("scaler_state"):
        scaler.load_state_dict(checkpoint["scaler_state"])
    return checkpoint


def get_padding_mask(tokens: torch.Tensor, pad_token_id: int) -> torch.Tensor:
    """计算填充位置掩码，供注意力层屏蔽填充标记。

    参数说明：
    - ``tokens``：来自数据加载器的输入序列张量。
    - ``pad_token_id``：分词器定义的填充标记索引。
    """
    return tokens.eq(pad_token_id)
