# Architecture Choice

This document explains the architecture used by ECI-SLM, why it was chosen, and what trade-offs it makes.

## Model Family

ECI-SLM is a decoder-only Transformer implemented in TensorFlow (`src/slm.py`), inspired by the nanochat style of compact, efficient language models.

It is built for a small-compute setting where training stability and iteration speed matter more than peak benchmark scores.

## Default Configuration

From `SLMConfig` / `main.py` defaults:

- `vocab_size = 8000`
- `block_size = 256`
- `n_layer = 6`
- `n_head = 6`
- `n_kv_head = 2` (GQA)
- `n_embd = 384`
- `dropout = 0.1` (train)
- `rope_theta = 100000.0`
- `tie_embeddings = True`

Trainable parameters (default): `12,509,184`.

## Core Building Blocks

### 1) Token embedding and tied LM head

- Input tokens go through a learned embedding table.
- Output logits reuse the same embedding matrix (`tie_embeddings=True`) to reduce parameter count and improve small-model efficiency.

### 2) Pre-norm residual blocks

Each block applies:

- RMSNorm -> Causal Self-Attention -> residual add
- RMSNorm -> ReLU^2 MLP -> residual add

This pre-norm pattern improves optimization stability at small scale.

### 3) Grouped-query attention (GQA)

- Query heads: `n_head`
- Key/Value heads: `n_kv_head`

With `6` query heads and `2` KV heads, KV is shared across query groups. This lowers memory and compute versus full multi-head KV without removing multi-head query behavior.

### 4) RoPE positional encoding

- Rotary embeddings are applied to Q and K.
- No learned absolute positional embedding table is used.

RoPE gives robust positional generalization for compact autoregressive models.

### 5) Bias-free linear layers + parameter-free RMSNorm

- Linear projections are bias-free.
- RMSNorm is parameter-free in this implementation.

This keeps the model compact and close to the intended minimalist design.

## Why This Architecture Was Chosen

- Small and trainable on limited hardware.
- Good fit for mixed corpus training (general + domain).
- Simple implementation path in TensorFlow with controllable stability.
- Strong enough to capture ECI procedural language after pretrain + SFT.

## Training-Oriented Design Decisions

The architecture is paired with pipeline controls in `main.py`:

- warmup cap to avoid long flatline starts,
- plateau-triggered LR scaling,
- early stopping on stalled validation,
- mixed precision options and distributed strategy support,
- checkpoint pruning with best-checkpoint retention.

These are not architecture layers, but they are part of the practical system design that makes this model usable.

## Known Trade-offs

- `block_size=256` limits long-context legal/procedural reasoning.
- Small parameter budget improves speed but reduces factual capacity.
- GQA and tied embeddings save compute/params but can cap expressiveness.
- ReLU^2 MLP is efficient, but not always as strong as larger gated MLP designs at equal token budgets.

## Not Implemented (By Design)

- No encoder or retrieval component.
- No KV cache optimization path for ultra-fast long generation.
- No RLHF or preference optimization stage.
- No external tool-use mechanism in model inference.

## Upgrade Path

If quality needs to improve while keeping the same architecture style:

1. Increase context window (`block_size`) first.
2. Improve data quality/dedup before scaling model size.
3. Scale depth/width conservatively (for example to 8 layers / 512 embd).
4. Re-evaluate domain/general mix ratio per run objective.

This preserves the current codebase shape while moving toward better factual and generative quality.
