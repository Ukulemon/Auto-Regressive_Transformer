"""Tokenization utilities for the autoregressive Transformer language model.

This module implements several tokenizers that can operate at the character,
word and subword level. Each tokenizer exposes a minimal interface with
``fit``/``encode``/``decode`` methods and maintains an internal vocabulary with
``stoi`` (string-to-index) and ``itos`` (index-to-string) mappings.  The
implementation is intentionally lightweight and does not depend on external
packages so that it runs out-of-the-box in restricted environments.

The subword tokenizer implements a simple byte pair encoding (BPE) algorithm
that is adequate for medium sized corpora.  The behaviour is similar to the
algorithm described in "Neural Machine Translation of Rare Words with
Subword Units" (Sennrich et al., 2016).
"""
from __future__ import annotations

import collections
import math
import re
from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence

import torch

__all__ = [
    "SpecialTokens",
    "BaseTokenizer",
    "CharacterTokenizer",
    "WordTokenizer",
    "BPETokenizer",
    "build_tokenizer",
]


@dataclass(frozen=True)
class SpecialTokens:
    """Container for commonly used special token strings."""

    pad: str = "<pad>"
    bos: str = "<bos>"
    eos: str = "<eos>"
    unk: str = "<unk>"


class BaseTokenizer:
    """Abstract base class for tokenizers."""

    def __init__(self, special_tokens: SpecialTokens | None = None) -> None:
        """初始化基础分词器，建立特殊符号词表。

        参数说明：
        - ``special_tokens``：来自调用方的特殊符号配置，默认为``SpecialTokens``类的默认值。
        """
        self.special_tokens = special_tokens or SpecialTokens()
        self.stoi: Dict[str, int] = {}
        self.itos: List[str] = []

    # ------------------------------------------------------------------
    # Vocabulary handling utilities
    # ------------------------------------------------------------------
    def _add_token_to_vocab(self, token: str) -> int:
        """将新标记加入词表，并返回对应索引。

        参数说明：
        - ``token``：来自语料或特殊符号的字符串标记。
        """
        if token not in self.stoi:
            self.stoi[token] = len(self.itos)
            self.itos.append(token)
        return self.stoi[token]

    def _init_base_vocab(self) -> None:
        """初始化基础词表，写入填充、开始、结束、未知等特殊符号。"""
        for token in (
            self.special_tokens.pad,
            self.special_tokens.bos,
            self.special_tokens.eos,
            self.special_tokens.unk,
        ):
            self._add_token_to_vocab(token)

    # ------------------------------------------------------------------
    # API surface
    # ------------------------------------------------------------------
    def fit(self, texts: Iterable[str]) -> None:
        """基类接口：根据输入语料构建词表。

        参数说明：
        - ``texts``：来自数据预处理阶段的文本迭代器。
        """
        raise NotImplementedError

    def encode(self, text: str, add_special_tokens: bool = True) -> List[int]:
        """基类接口：将文本转换为标记ID列表。"""
        raise NotImplementedError

    def decode(self, token_ids: Sequence[int], skip_special_tokens: bool = True) -> str:
        """基类接口：将标记ID序列还原为文本。"""
        raise NotImplementedError

    # ------------------------------------------------------------------
    # Helper methods
    # ------------------------------------------------------------------
    @property
    def pad_token_id(self) -> int:
        """返回填充标记在词表中的索引。"""
        return self.stoi[self.special_tokens.pad]

    @property
    def bos_token_id(self) -> int:
        """返回开始标记在词表中的索引。"""
        return self.stoi[self.special_tokens.bos]

    @property
    def eos_token_id(self) -> int:
        """返回结束标记在词表中的索引。"""
        return self.stoi[self.special_tokens.eos]

    @property
    def unk_token_id(self) -> int:
        """返回未知标记在词表中的索引。"""
        return self.stoi[self.special_tokens.unk]

    def vocab_size(self) -> int:
        """获取当前词表大小。"""
        return len(self.itos)


class CharacterTokenizer(BaseTokenizer):
    """Simple character level tokenizer."""

    def __init__(self, lowercase: bool = True, special_tokens: SpecialTokens | None = None) -> None:
        """初始化字符级分词器，可根据配置决定是否转小写。

        参数说明：
        - ``lowercase``：来自配置，控制是否将输入统一为小写。
        - ``special_tokens``：可选特殊符号设置。
        """
        super().__init__(special_tokens)
        self.lowercase = lowercase

    def fit(self, texts: Iterable[str]) -> None:
        """遍历语料中的字符构建字符词表。

        参数说明：
        - ``texts``：由数据加载流程提供的语料迭代器。
        """
        self._init_base_vocab()
        charset = set()
        for text in texts:
            if self.lowercase:
                text = text.lower()
            charset.update(text)
        for char in sorted(charset):
            if char.strip() == "":
                # preserve whitespace but normalise representation
                char = " "
            self._add_token_to_vocab(char)

    def encode(self, text: str, add_special_tokens: bool = True) -> List[int]:
        """将输入字符串逐字符映射为ID序列。

        参数说明：
        - ``text``：来自语料或推理阶段的原始文本。
        - ``add_special_tokens``：外部调用传入的布尔值，决定是否添加BOS/EOS标记。
        """
        if self.lowercase:
            text = text.lower()
        tokens = [self.stoi.get(ch, self.unk_token_id) for ch in text]
        if add_special_tokens:
            return [self.bos_token_id] + tokens + [self.eos_token_id]
        return tokens

    def decode(self, token_ids: Sequence[int], skip_special_tokens: bool = True) -> str:
        """根据ID序列还原字符文本。

        参数说明：
        - ``token_ids``：模型输出或数据集中读取的标记ID列表。
        - ``skip_special_tokens``：控制是否跳过特殊符号，由调用方设置。
        """
        chars: List[str] = []
        for token_id in token_ids:
            if skip_special_tokens and token_id in (
                self.pad_token_id,
                self.bos_token_id,
                self.eos_token_id,
            ):
                continue
            if 0 <= token_id < len(self.itos):
                chars.append(self.itos[token_id])
            else:
                chars.append(self.special_tokens.unk)
        return "".join(chars)


def _basic_tokenize(text: str) -> List[str]:
    """基础分词函数：去除标点并按空白切分单词。

    参数说明：
    - ``text``：来自预处理流程的原始字符串。
    返回清洗后的词列表。
    """
    cleaned = re.sub(r"[^\w\s]", " ", text)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned.split()


_EN_STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "but",
    "by",
    "for",
    "if",
    "in",
    "into",
    "is",
    "it",
    "no",
    "not",
    "of",
    "on",
    "or",
    "such",
    "that",
    "the",
    "their",
    "then",
    "there",
    "these",
    "they",
    "this",
    "to",
    "was",
    "will",
    "with",
}


class WordTokenizer(BaseTokenizer):
    """Whitespace + punctuation aware word tokenizer with stop word removal."""

    def __init__(
        self,
        lowercase: bool = True,
        stopwords: Iterable[str] | None = None,
        special_tokens: SpecialTokens | None = None,
    ) -> None:
        """初始化词级分词器，并配置停用词集合。

        参数说明：
        - ``lowercase``：来自配置，决定是否统一小写。
        - ``stopwords``：可以由外部传入的停用词列表，默认使用内置英文停用词。
        - ``special_tokens``：特殊符号配置。
        """
        super().__init__(special_tokens)
        self.lowercase = lowercase
        self.stopwords = set(stopwords) if stopwords is not None else set(_EN_STOPWORDS)

    def fit(self, texts: Iterable[str]) -> None:
        """统计语料中的词频并构建词级词表。

        参数说明：
        - ``texts``：来自数据预处理阶段的文本迭代器。
        """
        self._init_base_vocab()
        counter: Dict[str, int] = collections.Counter()
        for text in texts:
            if self.lowercase:
                text = text.lower()
            tokens = [token for token in _basic_tokenize(text) if token not in self.stopwords]
            counter.update(tokens)
        for token, _ in counter.most_common():
            self._add_token_to_vocab(token)

    def encode(self, text: str, add_special_tokens: bool = True) -> List[int]:
        """将文本按词切分并映射到词表索引。

        参数说明：
        - ``text``：来自语料或推理输入的字符串。
        - ``add_special_tokens``：调用方指定是否添加序列首尾符号。
        """
        if self.lowercase:
            text = text.lower()
        tokens = [token for token in _basic_tokenize(text) if token not in self.stopwords]
        ids = [self.stoi.get(tok, self.unk_token_id) for tok in tokens]
        if add_special_tokens:
            return [self.bos_token_id] + ids + [self.eos_token_id]
        return ids

    def decode(self, token_ids: Sequence[int], skip_special_tokens: bool = True) -> str:
        """根据词ID序列还原空格分隔的文本。

        参数说明：
        - ``token_ids``：模型输出或数据中存储的词索引列表。
        - ``skip_special_tokens``：由调用方决定是否移除特殊符号。
        """
        words: List[str] = []
        for token_id in token_ids:
            if skip_special_tokens and token_id in (
                self.pad_token_id,
                self.bos_token_id,
                self.eos_token_id,
            ):
                continue
            if 0 <= token_id < len(self.itos):
                words.append(self.itos[token_id])
            else:
                words.append(self.special_tokens.unk)
        return " ".join(words).strip()


class BPETokenizer(BaseTokenizer):
    """A tiny byte pair encoding tokenizer.

    The implementation keeps track of merges learned during ``fit``.  During
    encoding, the same merges are applied greedily.  The tokenizer operates on
    whitespace separated words and inserts a marker (``▁``) to indicate word
    boundaries.  This is similar to SentencePiece and helps the model reason
    about whitespace explicitly.
    """

    def __init__(
        self,
        vocab_size: int = 2000,
        lowercase: bool = True,
        min_frequency: int = 2,
        special_tokens: SpecialTokens | None = None,
    ) -> None:
        """初始化子词级BPE分词器，并配置词表规模等超参数。

        参数说明：
        - ``vocab_size``：来自配置，控制目标子词表大小。
        - ``lowercase``：是否统一小写。
        - ``min_frequency``：最小合并频次阈值。
        - ``special_tokens``：特殊符号配置。
        """
        super().__init__(special_tokens)
        self.vocab_size = vocab_size
        self.lowercase = lowercase
        self.min_frequency = min_frequency
        self.merges: Dict[tuple[str, str], str] = {}
        self.merge_ranks: Dict[tuple[str, str], int] = {}

    def fit(self, texts: Iterable[str]) -> None:
        """执行BPE训练流程，统计合并对并生成最终词表。

        参数说明：
        - ``texts``：数据预处理阶段提供的语料迭代器。
        """
        self._init_base_vocab()
        corpus: List[List[str]] = []
        counter: Dict[tuple[str, ...], int] = collections.Counter()
        for text in texts:
            if self.lowercase:
                text = text.lower()
            words = _basic_tokenize(text)
            if not words:
                continue
            processed = []
            for word in words:
                tokens = tuple(["▁"] + list(word))
                counter[tokens] += 1
                processed.append(tokens)
            corpus.append(processed)

        merges: Dict[tuple[str, str], str] = {}
        merge_ranks: Dict[tuple[str, str], int] = {}
        vocab = set()
        for tokens in counter:
            vocab.update(tokens)

        def get_stats(counter_map: Dict[tuple[str, ...], int]) -> Dict[tuple[str, str], int]:
            """内部函数：统计候选符号对的出现频次，用于选择合并对象。"""
            stats: Dict[tuple[str, str], int] = collections.Counter()
            for token_tuple, freq in counter_map.items():
                if freq < self.min_frequency:
                    continue
                for i in range(len(token_tuple) - 1):
                    stats[(token_tuple[i], token_tuple[i + 1])] += freq
            return stats

        merge_index = 0
        while len(vocab) < self.vocab_size:
            stats = get_stats(counter)
            if not stats:
                break
            best_pair, best_freq = max(stats.items(), key=lambda kv: kv[1])
            if best_freq < self.min_frequency:
                break
            merged_token = "".join(best_pair)
            merges[best_pair] = merged_token
            merge_ranks[best_pair] = merge_index
            merge_index += 1
            vocab.add(merged_token)

            new_counter: Dict[tuple[str, ...], int] = collections.Counter()
            for token_tuple, freq in counter.items():
                new_tokens: List[str] = []
                i = 0
                while i < len(token_tuple):
                    if i < len(token_tuple) - 1 and (token_tuple[i], token_tuple[i + 1]) == best_pair:
                        new_tokens.append(merged_token)
                        i += 2
                    else:
                        new_tokens.append(token_tuple[i])
                        i += 1
                new_counter[tuple(new_tokens)] += freq
            counter = new_counter

        for token in sorted(vocab):
            self._add_token_to_vocab(token)
        self.merges = merges
        self.merge_ranks = merge_ranks

    def _merge_word(self, word: str) -> List[str]:
        """将单词按已学习的BPE合并规则拆分成子词序列。

        参数说明：
        - ``word``：来自输入文本的原始单词字符串。
        返回应用合并后的符号列表。
        """
        if self.lowercase:
            word = word.lower()
        symbols: List[str] = ["▁"] + list(word)
        pairs = self._get_pairs(symbols)
        while pairs:
            ranked_pairs = [(self.merge_ranks.get(pair, float("inf")), pair) for pair in pairs]
            best_rank, pair = min(ranked_pairs, key=lambda item: item[0])
            if not math.isfinite(best_rank):
                break
            first, second = pair
            merged = self.merges[pair]
            new_symbols: List[str] = []
            i = 0
            while i < len(symbols):
                if i < len(symbols) - 1 and symbols[i] == first and symbols[i + 1] == second:
                    new_symbols.append(merged)
                    i += 2
                else:
                    new_symbols.append(symbols[i])
                    i += 1
            symbols = new_symbols
            pairs = self._get_pairs(symbols)
        return symbols

    @staticmethod
    def _get_pairs(symbols: Sequence[str]) -> List[tuple[str, str]]:
        """计算符号序列中相邻符号对的列表。"""
        pairs = []
        for i in range(len(symbols) - 1):
            pairs.append((symbols[i], symbols[i + 1]))
        return pairs

    def encode(self, text: str, add_special_tokens: bool = True) -> List[int]:
        """将文本拆分为子词并映射到词表索引。

        参数说明：
        - ``text``：来自语料或推理输入的字符串。
        - ``add_special_tokens``：决定是否添加序列首尾符，由调用方设定。
        """
        if self.lowercase:
            text = text.lower()
        words = _basic_tokenize(text)
        tokens: List[int] = []
        for word in words:
            for symbol in self._merge_word(word):
                tokens.append(self.stoi.get(symbol, self.unk_token_id))
        if add_special_tokens:
            tokens = [self.bos_token_id] + tokens + [self.eos_token_id]
        return tokens

    def decode(self, token_ids: Sequence[int], skip_special_tokens: bool = True) -> str:
        """根据子词ID序列还原可读文本。

        参数说明：
        - ``token_ids``：模型生成或数据集中保存的索引列表。
        - ``skip_special_tokens``：由调用方指定是否忽略特殊符号。
        """
        words: List[str] = []
        current_word: List[str] = []
        for token_id in token_ids:
            if skip_special_tokens and token_id in (
                self.pad_token_id,
                self.bos_token_id,
                self.eos_token_id,
            ):
                if token_id == self.eos_token_id and current_word:
                    words.append("".join(current_word))
                    current_word = []
                continue
            token = self.itos[token_id] if 0 <= token_id < len(self.itos) else self.special_tokens.unk
            if token.startswith("▁"):
                if current_word:
                    words.append("".join(current_word))
                current_word = [token[1:]]
            else:
                current_word.append(token)
        if current_word:
            words.append("".join(current_word))
        return " ".join(filter(None, words))


def build_tokenizer(
    tokenizer_type: str,
    **kwargs,
) -> BaseTokenizer:
    """根据类型字符串构建对应分词器实例。

    参数说明：
    - ``tokenizer_type``：来自配置或命令行的分词器类型标识。
    - ``kwargs``：透传给具体分词器构造函数的参数，通常来自用户配置。
    """
    tokenizer_type = tokenizer_type.lower()
    if tokenizer_type == "char":
        return CharacterTokenizer(**kwargs)
    if tokenizer_type == "word":
        return WordTokenizer(**kwargs)
    if tokenizer_type in {"bpe", "subword"}:
        return BPETokenizer(**kwargs)
    raise ValueError(f"Unsupported tokenizer type: {tokenizer_type}")


def pad_sequences(sequences: List[List[int]], pad_token_id: int) -> torch.Tensor:
    """将一批可变长序列填充为张量。

    参数说明：
    - ``sequences``：来自数据集或生成流程的标记ID列表集合。
    - ``pad_token_id``：分词器提供的填充标记索引，用于对齐长度。
    序列会统一填充至批次内的最大长度。
    """

    if not sequences:
        raise ValueError("Cannot pad empty sequence list")
    max_len = max(len(seq) for seq in sequences)
    padded = torch.full((len(sequences), max_len), pad_token_id, dtype=torch.long)
    for i, seq in enumerate(sequences):
        padded[i, : len(seq)] = torch.tensor(seq, dtype=torch.long)
    return padded
