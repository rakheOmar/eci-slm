# ECI-SLM

Docs: [Design Notes](docs/ARCHITECHTURE.md) | [Corpus Notes](docs/DATA.md) | [Model Behavior](docs/RESULT.md)

ECI-SLM is a compact TensorFlow decoder-only language model focused on Election Commission of India (ECI) procedures and public-election process text.

The project is designed around a practical goal: build a small, trainable model that can run on limited hardware while still learning domain language and producing usable ECI-style answers after SFT.

## What Is In This Repo

- A nanochat-inspired Transformer (`src/slm.py`) with RoPE, RMSNorm, ReLU^2 MLP, and grouped-query attention.
- A single pipeline entrypoint (`main.py`) for:
  - tokenizer training,
  - pretrain binary creation (mixed ECI + English),
  - supervised fine-tuning (assistant-only masked loss),
  - checkpointing and resume.
- Evaluation script (`src/eval.py`) that generates fixed prompt suites and saves CSV/JSON outputs.

## Training Flow

`main.py` supports:

- `--mode prepare`: build tokenizer + stage data.
- `--mode train`: train from prepared artifacts.
- `--mode prepare_and_train`: do both in one run.

Stages:

- `--stage pretrain`: mixed next-token LM training on `.bin` token streams.
- `--stage sft`: assistant-only masked SFT from Q/A text using `IGNORE_INDEX=-100`.

Key controls:

- Architecture: `--block_size --n_layer --n_head --n_kv_head --n_embd --untied_head`
- Data mix: `--english_ratio --mix_chunk_tokens --val_split`
- Optimization: `--learning_rate --warmup_steps --min_lr_frac --weight_decay`
- Stability: `--warmup_cap_frac --plateau_patience_evals --plateau_lr_decay --early_stop_patience_evals`
- Distribution: `--strategy auto|mirrored|single|cpu`
- Resume/init: `--resume --resume_step --init_checkpoint_dir --init_step`

## Repository Structure

```tree
eci-language-model/
├── main.py
├── pyproject.toml
├── README.md
│
├── src/
│   ├── slm.py
│   ├── sft.py
│   ├── eval.py
│   ├── checkpoint.py
│   ├── dataloader.py
│   └── tokenizer.py
│
├── data/
│   ├── pretrain/
│   ├── pretrain_expanded/
│   ├── pretrain_augmented/
│   ├── english_pretrain/
│   ├── instruct/
│   └── scripts/
│
├── notebooks/
│   ├── train.ipynb
│   └── old.ipynb
│
├── docs/
│   ├── DOCS.md
│   ├── RESEARCH.md
│   └── CHANGELOG.md
│
├── checkpoints/
├── artifact/
├── artifact_sft/
└── results/
```

## Known Limitations

- Context window is short (`256`), so long legal passages truncate aggressively.
- ECI-specific data is small relative to English background data.
- Some augmented corpora are repetitive, which can increase looped or template-like generations.
- Current results are mostly qualitative; no benchmark harness is checked in yet.

## License

No license file is currently included. Add one before public distribution.
