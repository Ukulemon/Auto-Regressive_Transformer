"""Model definitions for the autoregressive Transformer language model."""
from __future__ import annotations

import math
from functools import partial
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding."""

    def __init__(self, d_model: int, max_len: int = 10000, dropout: float = 0.1) -> None:
        """构造位置编码模块，预生成正弦/余弦编码供后续叠加。

        参数说明：
        - ``d_model``：来自模型配置的隐藏维度大小。
        - ``max_len``：根据最大序列长度配置确定的预生成长度。
        - ``dropout``：由训练配置传入的随机失活比例。
        """
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """将预生成的位置编码叠加到输入表示上，并应用Dropout稳定训练。

        参数说明：
        - ``x``：来自词嵌入层或上一层输出的张量，形状为``(batch, seq_len, d_model)``。
        """
        x = x + self.pe[:, : x.size(1)]
        return self.dropout(x)


class AutoregressiveTransformer(nn.Module):
    """Decoder-only Transformer language model."""

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 512,
        nhead: int = 8,
        num_layers: int = 6,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        max_seq_len: int = 1024,
        activation: str = "gelu",
        gradient_checkpointing: bool = False,
    ) -> None:
        """构造自回归Transformer模型，并初始化所有核心子模块。

        参数说明：
        - ``vocab_size``：来自分词器``vocab_size``方法的词表大小。
        - ``d_model``/``nhead``/``num_layers``/``dim_feedforward``/``dropout``/``activation``：来自训练配置或命令行的模型超参数。
        - ``max_seq_len``：依据序列长度配置确定的位置编码范围。
        - ``gradient_checkpointing``：由配置控制是否启用梯度检查点节省显存。
        """
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model, max_seq_len, dropout=dropout)
        self.layers = nn.ModuleList(
            [
                nn.TransformerEncoderLayer(
                    d_model=d_model,
                    nhead=nhead,
                    dim_feedforward=dim_feedforward,
                    dropout=dropout,
                    activation=activation,
                    batch_first=True,
                    norm_first=True,
                )
                for _ in range(num_layers)
            ]
        )
        self.norm = nn.LayerNorm(d_model)
        self.output_head = nn.Linear(d_model, vocab_size)
        self.gradient_checkpointing = gradient_checkpointing

        self._reset_parameters()

    def _reset_parameters(self) -> None:
        """按照Transformer惯例初始化嵌入层与输出层的权重。"""
        nn.init.normal_(self.token_embedding.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.output_head.bias)
        nn.init.normal_(self.output_head.weight, mean=0.0, std=0.02)

    def forward(self, tokens: torch.Tensor, padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """执行前向传播并返回下一标记预测的logits。

        参数说明：
        - ``tokens``：由``DataLoader``批量提供的输入ID序列张量。
        - ``padding_mask``：通过``get_padding_mask``构造的填充位置掩码，可来自训练或评估阶段。
        返回值为形状``(batch, seq_len, vocab_size)``的分类logits。
        """

        # tokens: (batch, seq_len)
        device = tokens.device
        embeddings = self.token_embedding(tokens)
        hidden = self.positional_encoding(embeddings)
        seq_len = tokens.size(1)
        causal_mask = torch.triu(
            torch.ones((seq_len, seq_len), device=device, dtype=torch.bool),
            diagonal=1,
        )

        def layer_forward(layer: nn.Module, hidden_states: torch.Tensor) -> torch.Tensor:
            """内部函数：封装单层自注意力块的调用，供梯度检查点复用。"""
            return layer(
                hidden_states,
                src_mask=causal_mask,
                src_key_padding_mask=padding_mask,
            )

        for layer in self.layers:
            if self.gradient_checkpointing and self.training:
                hidden = checkpoint(partial(layer_forward, layer), hidden)
            else:
                hidden = layer_forward(layer, hidden)
        hidden = self.norm(hidden)
        logits = self.output_head(hidden)
        return logits

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_length: int = 100,
        eos_token_id: Optional[int] = None,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
    ) -> torch.Tensor:
        """基于给定提示自回归生成文本序列。

        参数说明：
        - ``input_ids``：来自分词器对提示语编码后的张量，通常由外部调用构造。
        - ``max_length``：由配置设定的最大生成步数。
        - ``eos_token_id``：来自分词器的结束符ID，用于提前终止生成。
        - ``temperature``/``top_k``：生成策略相关的采样超参数，来自配置或命令行输入。
        返回包含原始提示及新增标记的完整序列张量。
        """
        self.eval()
        generated = input_ids.clone()
        device = generated.device
        for _ in range(max_length):
            logits = self.forward(generated)
            next_token_logits = logits[:, -1, :]
            if temperature != 1.0:
                next_token_logits = next_token_logits / temperature
            if top_k is not None and top_k > 0:
                values, _ = torch.topk(next_token_logits, top_k)
                min_values = values[:, -1].unsqueeze(-1)
                next_token_logits = torch.where(
                    next_token_logits < min_values,
                    torch.full_like(next_token_logits, float("-inf")),
                    next_token_logits,
                )
            probs = F.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            generated = torch.cat([generated, next_token], dim=1)
            if eos_token_id is not None and (next_token == eos_token_id).all():
                break
        return generated
