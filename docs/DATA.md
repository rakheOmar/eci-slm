# Data Documentation

This document describes the data used by ECI-SLM: where it comes from, how it is organized, and the token composition used for training.

## Directory Layout

- `data/pretrain/`: core ECI-domain reference text (manuals, guides, overviews).
- `data/pretrain_expanded/`: FAQ-derived expanded text generated from instruct material.
- `data/pretrain_augmented/`: paraphrased/augmented variants of core ECI text.
- `data/english_pretrain/`: general English background corpus (`fineweb_*`).
- `data/instruct/`: supervised Q/A style text used for SFT.

## Sources

### ECI-domain corpora

Primarily policy/procedure-oriented documents represented in files such as:

- returning officer SOP references,
- candidate/observer/police handbooks,
- model code of conduct and election workflow guides,
- election FAQ-style Q/A files,
- high-level ECI and Wikipedia summary text.

These are intended to teach procedural language and election-domain vocabulary.

### General English corpus

- `data/english_pretrain/fineweb_cc_6p6m.txt`
- `data/english_pretrain/fineweb_sample.txt`

These files provide broader language coverage and fluency support.

## Token Counts

Counts below were produced with `python data/scripts/count_tokens.py` using the project SentencePiece tokenizer (`artifact/eci_slm_tokenizer.model`).

| Directory | Files | Tokens |
|---|---:|---:|
| `data/pretrain` | 11 | 521,037 |
| `data/pretrain_expanded` | 16 | 245,313 |
| `data/pretrain_augmented` | 11 | 972,779 |
| `data/english_pretrain` | 2 | 9,540,409 |
| `data/instruct` | 16 | 60,563 |
| **Grand total** | **56** | **11,340,101** |

## Pretrain Mix Reality

The pretrain stage uses only:

- ECI pool = `pretrain + pretrain_expanded + pretrain_augmented`
- English pool = `english_pretrain`

Token pool totals for pretraining:

- ECI pool: `1,739,129`
- English pool: `9,540,409`
- Combined pretrain pool: `11,279,538`

With default `--english_ratio 0.95`, effective mixed tokens are capped by available English tokens:

- target ECI tokens: ~`502,126`
- target English tokens: ~`9,540,408`
- mixed total: ~`10,042,534`

So even though total ECI pool is larger than 5% of the raw pool, the default mix intentionally keeps ECI as a small but persistent domain signal.

## SFT Data Format

`data/instruct/*.txt` is parsed by `src/sft.py` using patterns like:

- `User: ... Assistant: ... <END>`
- `Question: ... Answer: ... <END>`

During SFT example creation:

- prompt tokens are included in input `x`,
- loss is masked on prompt tokens,
- only assistant answer tokens (plus EOS) contribute to target `y`.

This avoids teaching the model to imitate the user side of the conversation.

## Data Quality Notes

- Some augmented files are repetitive and bureaucratic in style; this can improve domain recall but hurt generation diversity.
- FAQ-derived expanded data helps question-answer alignment, but can bias toward formulaic response templates.
- English corpus is much larger, so without careful ratio settings, general style can dominate domain behavior.

## Reproducibility

To regenerate counts:

```bash
python data/scripts/count_tokens.py
```

To rebuild pretrain bins with a custom ratio:

```bash
python main.py --mode prepare --stage pretrain --english_ratio 0.90 --rebuild_bins
```

To rebuild SFT arrays:

```bash
python main.py --mode prepare --stage sft --rebuild_bins
```

## Compliance Note

Before redistribution or production use, verify data licensing and attribution requirements for every upstream source included in `data/`.
