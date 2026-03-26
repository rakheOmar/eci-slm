#!/usr/bin/env python3
"""ECI-SLM model (Nanochat-inspired, TensorFlow).

Implemented features:
- RoPE (no positional embeddings)
- QK RMS normalization
- RMSNorm pre-norm blocks
- ReLU^2 MLP
- Untied token embedding and LM head
- No bias in linear layers
- Optional GQA (n_kv_head <= n_head)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import cast

import tensorflow as tf


@dataclass
class SLMConfig:
    vocab_size: int = 32000
    block_size: int = 2048
    n_layer: int = 12
    n_head: int = 12
    n_kv_head: int | None = None
    n_embd: int = 768
    dropout: float = 0.1
    rms_norm_eps: float = 1e-6
    rope_theta: float = 100000.0

    def __post_init__(self) -> None:
        if self.n_kv_head is None:
            self.n_kv_head = int(self.n_head)
        kv_heads = int(self.n_kv_head)
        if self.n_embd % self.n_head != 0:
            raise ValueError(
                f"n_embd ({self.n_embd}) must be divisible by n_head ({self.n_head})"
            )
        if self.n_head % kv_heads != 0:
            raise ValueError(
                f"n_head ({self.n_head}) must be divisible by n_kv_head ({kv_heads})"
            )
        head_dim = self.n_embd // self.n_head
        if head_dim % 2 != 0:
            raise ValueError(f"head_dim must be even for RoPE, got {head_dim}")


def rms_norm(x: tf.Tensor, eps: float = 1e-6) -> tf.Tensor:
    return x * tf.math.rsqrt(tf.reduce_mean(tf.square(x), axis=-1, keepdims=True) + eps)


class RMSNorm(tf.keras.layers.Layer):
    """Parameter-free RMSNorm to mirror nanochat's functional norm."""

    def __init__(self, eps: float = 1e-6, **kwargs):
        super().__init__(**kwargs)
        self.eps = eps

    def call(self, x: tf.Tensor) -> tf.Tensor:
        return rms_norm(x, self.eps)


class Linear(tf.keras.layers.Layer):
    """Bias-free linear layer."""

    def __init__(self, out_dim: int, **kwargs):
        super().__init__(**kwargs)
        self.out_dim = out_dim

    def build(self, input_shape):
        in_dim = int(input_shape[-1])
        self.kernel = self.add_weight(
            name="kernel",
            shape=(in_dim, self.out_dim),
            initializer=tf.keras.initializers.RandomNormal(stddev=0.02),
            trainable=True,
        )
        super().build(input_shape)

    def call(self, x: tf.Tensor) -> tf.Tensor:
        return tf.matmul(x, self.kernel)


class RotaryEmbedding(tf.keras.layers.Layer):
    def __init__(
        self, head_dim: int, max_seq_len: int, base: float = 100000.0, **kwargs
    ):
        super().__init__(**kwargs)
        self.head_dim = head_dim
        self.max_seq_len = max_seq_len
        self.base = base

    def build(self, input_shape):
        half = self.head_dim // 2
        idx = tf.range(0, half, dtype=tf.float32)
        inv_freq = 1.0 / (self.base ** (idx / tf.cast(half, tf.float32)))
        pos = tf.range(0, self.max_seq_len, dtype=tf.float32)
        freqs = tf.einsum("i,j->ij", pos, inv_freq)  # [T, D/2]
        self.cos = tf.Variable(tf.cos(freqs), trainable=False, name="rope_cos")
        self.sin = tf.Variable(tf.sin(freqs), trainable=False, name="rope_sin")
        super().build(input_shape)

    def call(self, x: tf.Tensor) -> tf.Tensor:
        # x: [B, T, H, D]
        t = tf.shape(x)[1]
        cos = self.cos[:t][tf.newaxis, :, tf.newaxis, :]  # [1,T,1,D/2]
        sin = self.sin[:t][tf.newaxis, :, tf.newaxis, :]
        cos = tf.cast(cos, x.dtype)
        sin = tf.cast(sin, x.dtype)

        x1, x2 = tf.split(x, num_or_size_splits=2, axis=-1)
        y1 = x1 * cos + x2 * sin
        y2 = -x1 * sin + x2 * cos
        return tf.concat([y1, y2], axis=-1)


class CausalSelfAttention(tf.keras.layers.Layer):
    def __init__(self, cfg: SLMConfig, **kwargs):
        super().__init__(**kwargs)
        self.n_head = cfg.n_head
        if cfg.n_kv_head is None:
            raise ValueError("n_kv_head must not be None after config validation")
        self.n_kv_head = int(cfg.n_kv_head)
        self.n_embd = cfg.n_embd
        self.head_dim = cfg.n_embd // cfg.n_head
        self.block_size = cfg.block_size
        self.group_size = self.n_head // self.n_kv_head

        self.q_proj = Linear(self.n_head * self.head_dim)
        self.k_proj = Linear(self.n_kv_head * self.head_dim)
        self.v_proj = Linear(self.n_kv_head * self.head_dim)
        self.o_proj = Linear(self.n_embd)

        self.qk_norm = RMSNorm(cfg.rms_norm_eps)
        self.attn_dropout = tf.keras.layers.Dropout(cfg.dropout)
        self.resid_dropout = tf.keras.layers.Dropout(cfg.dropout)
        self.rope = RotaryEmbedding(self.head_dim, cfg.block_size, cfg.rope_theta)

    def build(self, input_shape):
        mask = tf.linalg.band_part(
            tf.ones((self.block_size, self.block_size), dtype=tf.float32), -1, 0
        )
        self.causal_mask = tf.Variable(mask, trainable=False, name="causal_mask")
        super().build(input_shape)

    def _expand_kv_for_gqa(self, x: tf.Tensor) -> tf.Tensor:
        # x: [B, T, H_kv, D] -> [B, T, H_q, D]
        if self.n_kv_head == self.n_head:
            return x
        x = tf.repeat(x, repeats=self.group_size, axis=2)
        return x

    def call(self, x: tf.Tensor, training: bool = False) -> tf.Tensor:
        b = tf.shape(x)[0]
        t = tf.shape(x)[1]

        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        q = tf.reshape(q, (b, t, self.n_head, self.head_dim))
        k = tf.reshape(k, (b, t, self.n_kv_head, self.head_dim))
        v = tf.reshape(v, (b, t, self.n_kv_head, self.head_dim))

        q = self.rope(q)
        k = self.rope(k)
        q = self.qk_norm(q)
        k = self.qk_norm(k)

        k = self._expand_kv_for_gqa(k)
        v = self._expand_kv_for_gqa(v)

        q = tf.transpose(q, [0, 2, 1, 3])
        k = tf.transpose(k, [0, 2, 1, 3])
        v = tf.transpose(v, [0, 2, 1, 3])

        scale = tf.math.rsqrt(tf.cast(self.head_dim, q.dtype))
        scores = tf.matmul(q, k, transpose_b=True) * scale

        mask = self.causal_mask[:t, :t][tf.newaxis, tf.newaxis, :, :]
        # Use an fp16-safe mask value to avoid overflow/cast warnings on mixed precision.
        mask_val = tf.cast(-1e4, scores.dtype)
        scores = tf.where(mask > 0, scores, mask_val)

        # Softmax in fp32 for mixed-precision stability.
        scores_f32 = tf.cast(scores, tf.float32)
        probs = tf.cast(tf.nn.softmax(scores_f32, axis=-1), scores.dtype)
        probs = self.attn_dropout(probs, training=training)

        y = tf.matmul(probs, v)
        y = tf.transpose(y, [0, 2, 1, 3])
        y = tf.reshape(y, (b, t, self.n_embd))
        y = self.o_proj(y)
        y = self.resid_dropout(y, training=training)
        return y


class MLP(tf.keras.layers.Layer):
    def __init__(self, cfg: SLMConfig, **kwargs):
        super().__init__(**kwargs)
        hidden = 4 * cfg.n_embd
        self.fc = Linear(hidden)
        self.proj = Linear(cfg.n_embd)

    def call(self, x: tf.Tensor) -> tf.Tensor:
        x = self.fc(x)
        x = tf.square(tf.nn.relu(x))
        x = self.proj(x)
        return x


class Block(tf.keras.layers.Layer):
    def __init__(self, cfg: SLMConfig, **kwargs):
        super().__init__(**kwargs)
        self.norm1 = RMSNorm(cfg.rms_norm_eps)
        self.attn = CausalSelfAttention(cfg)
        self.norm2 = RMSNorm(cfg.rms_norm_eps)
        self.mlp = MLP(cfg)
        self.dropout = tf.keras.layers.Dropout(cfg.dropout)

    def call(self, x: tf.Tensor, training: bool = False) -> tf.Tensor:
        x = x + self.attn(self.norm1(x), training=training)
        x = x + self.dropout(self.mlp(self.norm2(x)), training=training)
        return x


class ECISLM(tf.keras.Model):
    def __init__(self, cfg: SLMConfig, **kwargs):
        super().__init__(**kwargs)
        self.config = cfg

        self.token_embed = tf.keras.layers.Embedding(
            input_dim=cfg.vocab_size,
            output_dim=cfg.n_embd,
            embeddings_initializer=tf.keras.initializers.RandomNormal(stddev=0.02),
        )
        self.embed_norm = RMSNorm(cfg.rms_norm_eps)
        self.embed_dropout = tf.keras.layers.Dropout(cfg.dropout)

        self.blocks = [Block(cfg) for _ in range(cfg.n_layer)]
        self.final_norm = RMSNorm(cfg.rms_norm_eps)
        self.lm_head = Linear(cfg.vocab_size)  # untied

    def call(
        self,
        inputs: tf.Tensor,
        training: bool | None = None,
        mask=None,
        targets: tf.Tensor | None = None,
    ):
        training = bool(training)
        x = self.token_embed(inputs)
        x = self.embed_norm(x)
        x = self.embed_dropout(x, training=training)

        for block in self.blocks:
            x = block(x, training=training)

        x = self.final_norm(x)
        logits = self.lm_head(x)

        if targets is None:
            return logits, None

        loss = tf.reduce_mean(
            tf.keras.losses.sparse_categorical_crossentropy(
                targets, tf.cast(logits, tf.float32), from_logits=True
            )
        )
        return logits, loss

    def generate(
        self,
        idx: tf.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: int | None = None,
    ) -> tf.Tensor:
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.config.block_size :]
            logits, _ = self.call(idx_cond, training=False)
            logits = logits[:, -1, :] / temperature

            if top_k is not None:
                k = tf.minimum(top_k, tf.shape(logits)[-1])
                values, _ = tf.math.top_k(logits, k=k)
                threshold = values[:, -1:]
                logits = tf.where(
                    logits < threshold, tf.constant(-1e4, logits.dtype), logits
                )

            probs = tf.nn.softmax(logits, axis=-1)
            next_token = tf.random.categorical(tf.math.log(probs + 1e-9), num_samples=1)
            next_token = tf.cast(next_token, dtype=idx.dtype)
            idx = tf.concat([idx, next_token], axis=1)
        return idx


def create_slm_model(config: SLMConfig | None = None) -> ECISLM:
    return cast(ECISLM, ECISLM(config or SLMConfig()))


def count_parameters(model: tf.keras.Model) -> int:
    return int(model.count_params())


if __name__ == "__main__":
    cfg = SLMConfig(vocab_size=32000, block_size=512, n_layer=4, n_head=4, n_embd=256)
    model = create_slm_model(cfg)
    _ = model(tf.zeros((1, cfg.block_size), dtype=tf.int32))
    model.summary()
    print(f"\nTotal parameters: {count_parameters(model):,}")
