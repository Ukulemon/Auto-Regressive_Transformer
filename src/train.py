"""Command line script for training and generating text with the Transformer model."""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Tuple

import torch
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

from .data import create_dataloaders, preprocess_corpus, read_corpus
from .model import AutoregressiveTransformer
from .tokenizer import BaseTokenizer
from .utils import TrainingConfig, get_padding_mask, load_checkpoint, save_checkpoint


def parse_args() -> argparse.Namespace:
    """解析命令行参数，获取数据路径与模型训练相关配置。"""
    parser = argparse.ArgumentParser(description="Train an autoregressive Transformer language model")
    parser.add_argument("--data-path", type=str, required=True, help="Path to the training text file")
    parser.add_argument("--tokenizer", type=str, default="word", choices=["char", "word", "bpe", "subword"], help="Tokenizer type")
    parser.add_argument("--sequence-length", type=int, default=128, help="Training sequence length")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size")
    parser.add_argument("--d-model", type=int, default=512, help="Transformer hidden size")
    parser.add_argument("--nhead", type=int, default=8, help="Number of attention heads")
    parser.add_argument("--num-layers", type=int, default=6, help="Number of decoder layers")
    parser.add_argument("--dim-feedforward", type=int, default=2048, help="Feedforward dimension")
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--patience", type=int, default=3, help="Early stopping patience")
    parser.add_argument("--cosine-t-max", type=int, default=10, help="T_max for cosine annealing scheduler")
    parser.add_argument("--min-lr", type=float, default=1e-5)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--gradient-checkpointing", action="store_true")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume from")
    parser.add_argument("--pretrained", type=str, default=None, help="Path to pretrained weights for fine-tuning")
    parser.add_argument("--output-dir", type=str, default="checkpoints")
    parser.add_argument("--max-generate-length", type=int, default=200)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=0)
    parser.add_argument("--no-mixed-precision", action="store_true", help="Disable mixed precision training")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--vocab-size", type=int, default=2000, help="Vocabulary size for subword tokenizer")
    parser.add_argument("--min-frequency", type=int, default=2, help="Minimum frequency for subword merges")
    parser.add_argument("--generate", type=str, default=None, help="Prompt text to generate from after training")
    return parser.parse_args()


def setup_training(args: argparse.Namespace) -> Tuple[TrainingConfig, BaseTokenizer, torch.device, list[int]]:
    """根据命令行参数构建训练配置、分词器及训练数据。

    参数说明：
    - ``args``：由``parse_args``解析得到的命令行参数对象。
    返回训练配置、分词器实例、运行设备以及整段语料编码后的ID列表。
    """
    config = TrainingConfig(
        data_path=args.data_path,
        tokenizer=args.tokenizer,
        sequence_length=args.sequence_length,
        batch_size=args.batch_size,
        d_model=args.d_model,
        nhead=args.nhead,
        num_layers=args.num_layers,
        dim_feedforward=args.dim_feedforward,
        dropout=args.dropout,
        max_seq_len=max(args.sequence_length, 1024),
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        epochs=args.epochs,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        max_grad_norm=args.max_grad_norm,
        patience=args.patience,
        cosine_t_max=args.cosine_t_max,
        min_lr=args.min_lr,
        num_workers=args.num_workers,
        gradient_checkpointing=args.gradient_checkpointing,
        resume_from=args.resume,
        pretrained_weights=args.pretrained,
        output_dir=args.output_dir,
        max_generate_length=args.max_generate_length,
        temperature=args.temperature,
        top_k=args.top_k if args.top_k > 0 else None,
        mixed_precision=not args.no_mixed_precision,
    )

    raw_text = read_corpus(args.data_path)
    print(raw_text)
    tokenizer, token_ids = preprocess_corpus(
        raw_text,
        tokenizer_type=args.tokenizer,
        lowercase=True,
        vocab_size=args.vocab_size,
        min_frequency=args.min_frequency,
    )

    device = torch.device(args.device)

    return config, tokenizer, device, token_ids


def train_epoch(
    model: AutoregressiveTransformer,
    dataloader,
    optimizer,
    scheduler,
    scaler: GradScaler,
    device: torch.device,
    pad_token_id: int,
    accumulation_steps: int,
    max_grad_norm: float,
    mixed_precision: bool,
) -> float:
    """执行一次训练轮次，支持梯度累积与混合精度。"

    参数说明：
    - ``model``：正在训练的Transformer模型实例。
    - ``dataloader``：由``create_dataloaders``生成的训练数据迭代器。
    - ``optimizer``/``scheduler``：训练循环中的优化器与学习率调度器。
    - ``scaler``：来自``torch.cuda.amp``的缩放器，用于混合精度。
    - ``device``：训练设备，由``setup_training``确定。
    - ``pad_token_id``：分词器提供的填充标记索引。
    - ``accumulation_steps``：配置指定的梯度累积步数。
    - ``max_grad_norm``：梯度裁剪阈值，来自配置。
    - ``mixed_precision``：布尔值，表示是否启用混合精度，由命令行控制。
    返回本轮训练的平均损失。
    """
    model.train()
    optimizer.zero_grad(set_to_none=True)
    total_loss = 0.0
    num_batches = 0
    for step, (inputs, targets) in enumerate(tqdm(dataloader, desc="Training", leave=False)):
        inputs = inputs.to(device)
        targets = targets.to(device)
        padding_mask = get_padding_mask(inputs, pad_token_id)
        with autocast(enabled=mixed_precision):
            logits = model(inputs, padding_mask=padding_mask)
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                ignore_index=pad_token_id,
            )
            loss = loss / accumulation_steps
        scaler.scale(loss).backward()
        total_loss += loss.item()
        num_batches += 1
        if (step + 1) % accumulation_steps == 0:
            if max_grad_norm is not None:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            if scheduler is not None:
                scheduler.step()

    if num_batches % accumulation_steps != 0:
        if max_grad_norm is not None:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)
        if scheduler is not None:
            scheduler.step()

    avg_loss = (total_loss / max(1, num_batches)) * accumulation_steps
    return avg_loss


def evaluate(
    model: AutoregressiveTransformer,
    dataloader,
    device: torch.device,
    pad_token_id: int,
) -> float:
    """在验证集上评估模型性能，返回平均交叉熵损失。

    参数说明：
    - ``model``：待评估的Transformer模型。
    - ``dataloader``：验证数据迭代器，由``create_dataloaders``提供。
    - ``device``：运行设备。
    - ``pad_token_id``：分词器提供的填充标记索引。
    """
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for inputs, targets in tqdm(dataloader, desc="Evaluating", leave=False):
            inputs = inputs.to(device)
            targets = targets.to(device)
            padding_mask = get_padding_mask(inputs, pad_token_id)
            logits = model(inputs, padding_mask=padding_mask)
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                ignore_index=pad_token_id,
            )
            total_loss += loss.item()
    return total_loss / max(1, len(dataloader))


def main():
    """训练入口：解析参数、准备数据、执行训练并可选生成文本。"""
    args = parse_args()
    config, tokenizer, device, token_ids = setup_training(args)

    train_loader, val_loader = create_dataloaders(
        token_ids,
        tokenizer,
        batch_size=config.batch_size,
        sequence_length=config.sequence_length,
        num_workers=config.num_workers,
    )

    model = AutoregressiveTransformer(
        vocab_size=tokenizer.vocab_size(),
        d_model=config.d_model,
        nhead=config.nhead,
        num_layers=config.num_layers,
        dim_feedforward=config.dim_feedforward,
        dropout=config.dropout,
        max_seq_len=config.max_seq_len,
        gradient_checkpointing=config.gradient_checkpointing,
    ).to(device)

    optimizer = Adam(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    steps_per_epoch = max(1, len(train_loader) // max(1, config.gradient_accumulation_steps))
    scheduler = CosineAnnealingLR(optimizer, T_max=max(1, config.cosine_t_max * steps_per_epoch), eta_min=config.min_lr)
    scaler = GradScaler(enabled=config.mixed_precision and device.type == "cuda")

    start_epoch = 0
    best_val_loss = float("inf")
    patience_counter = 0

    if config.pretrained_weights:
        checkpoint = torch.load(config.pretrained_weights, map_location=device)
        model.load_state_dict(checkpoint["model_state"] if "model_state" in checkpoint else checkpoint)
        print(f"Loaded pretrained weights from {config.pretrained_weights}")

    if config.resume_from:
        state = load_checkpoint(model, optimizer, scheduler, scaler, config.resume_from, device)
        start_epoch = state.get("epoch", 0) + 1
        best_val_loss = state.get("best_val_loss", float("inf"))
        print(f"Resumed training from epoch {start_epoch}")

    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(start_epoch, config.epochs):
        print(f"Epoch {epoch + 1}/{config.epochs}")
        train_loss = train_epoch(
            model,
            train_loader,
            optimizer,
            scheduler,
            scaler,
            device,
            tokenizer.pad_token_id,
            config.gradient_accumulation_steps,
            config.max_grad_norm,
            config.mixed_precision and device.type == "cuda",
        )
        val_loss = evaluate(model, val_loader, device, tokenizer.pad_token_id)
        print(f"Train loss: {train_loss:.4f} | Val loss: {val_loss:.4f}")

        checkpoint_path = output_dir / f"epoch_{epoch + 1}.pt"
        save_checkpoint(
            model,
            optimizer,
            scheduler,
            scaler if scaler.is_enabled() else None,
            epoch,
            best_val_loss,
            config,
            checkpoint_path,
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            save_checkpoint(
                model,
                optimizer,
                scheduler,
                scaler if scaler.is_enabled() else None,
                epoch,
                best_val_loss,
                config,
                output_dir / "best.pt",
            )
        else:
            patience_counter += 1
            if patience_counter >= config.patience:
                print("Early stopping triggered")
                break

    if args.generate:
        prompt_ids = tokenizer.encode(args.generate, add_special_tokens=True)
        input_tensor = torch.tensor(prompt_ids, dtype=torch.long, device=device).unsqueeze(0)
        generated = model.generate(
            input_tensor,
            max_length=config.max_generate_length,
            eos_token_id=tokenizer.eos_token_id,
            temperature=config.temperature,
            top_k=config.top_k,
        )
        text = tokenizer.decode(generated[0].tolist())
        print("Generated text:\n", text)


if __name__ == "__main__":
    main()
