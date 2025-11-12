# Auto-Regressive Transformer

This project provides a configurable PyTorch implementation of an autoregressive
Transformer language model that can be trained on RTX 4060 class GPUs. The
package includes utilities for dataset preparation, tokenisation (character,
word and byte-pair/subword levels), mixed precision training, gradient
accumulation, checkpointing and text generation.

## Features

- Decoder-only Transformer implemented with `nn.TransformerDecoderLayer`
- Sinusoidal positional encoding, multi-head self-attention and feed-forward
  sublayers
- Character-, word- and BPE/subword-level tokenisers with stop-word removal for
  word-level tokenisation
- Automatic train/validation split (90/10) and sequence dataset construction
- Gradient accumulation, Adam optimiser, cosine annealing learning rate
  scheduler and optional gradient checkpointing
- Mixed precision (FP16) training through `torch.cuda.amp`
- Checkpoint saving & resuming, early stopping and support for fine-tuning from
  pretrained weights
- Greedy/top-k sampling based text generation with configurable temperature and
  stop conditions

## Project Structure

```
├── src/
│   ├── __init__.py
│   ├── data.py              # Dataset preparation utilities
│   ├── model.py             # Transformer model definition
│   ├── tokenizer.py         # Tokeniser implementations
│   ├── train.py             # Training and generation CLI
│   └── utils.py             # Config and checkpoint helpers
└── README.md
```

## Usage

### 1. Prepare data

Place a UTF-8 encoded text file on disk. The file can contain arbitrary text
(e.g. novels, articles or domain-specific corpora).

### 2. Train the model

Run the training script. The example below trains a word-level model for five
epochs with gradient accumulation and cosine annealing.

```bash
python -m src.train \
  --data-path /path/to/corpus.txt \
  --tokenizer word \
  --sequence-length 256 \
  --batch-size 8 \
  --epochs 5 \
  --gradient-accumulation-steps 4 \
  --d-model 512 \
  --nhead 8 \
  --num-layers 6 \
  --dim-feedforward 2048 \
  --cosine-t-max 5 \
  --output-dir checkpoints \
  --device cuda
```

Key options:

- `--tokenizer`: `char`, `word`, `bpe`/`subword`
- `--gradient-checkpointing`: enable transformer layer checkpointing to save
  memory
- `--no-mixed-precision`: disable FP16 if desired
- `--resume`: resume from a previously saved checkpoint
- `--pretrained`: load model weights for fine-tuning

Checkpoints are saved per epoch (including optimizer, scheduler and AMP scaler
state) under the output directory together with a `best.pt` file containing the
best validation checkpoint.

### 3. Generate text

After training, optionally generate text conditioned on a prompt:

```bash
python -m src.train \
  --data-path /path/to/corpus.txt \
  --tokenizer word \
  --sequence-length 256 \
  --batch-size 8 \
  --epochs 0 \
  --resume checkpoints/best.pt \
  --generate "Once upon a time" \
  --max-generate-length 100 \
  --temperature 0.8 \
  --top-k 50
```

The script loads the checkpoint, encodes the prompt and samples additional
tokens until an end-of-sequence token is produced or the maximum length is
reached.

## Requirements

- Python 3.9+
- PyTorch 2.0+

Install dependencies via:

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install tqdm
```

(Adjust the CUDA wheel index to match your local environment.)

## Notes

- The implementation targets RTX 4060 GPUs but runs on CPU for experimentation.
- Mixed precision and gradient checkpointing significantly reduce memory usage
  and are recommended for longer sequence lengths.
- For best results with the BPE tokenizer, increase `--vocab-size` according to
your corpus size.
