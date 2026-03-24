# Changelog

> All notable changes to the ECI-SLM project will be documented in this file.

---

## [Unreleased]

### Added

- **Data Collection**
  - ECI-specific pretraining data (~1.6M tokens across multiple documents)
  - Instruction/FAQ data for SFT (~56K tokens)
  - Expanded Q&A data (Q&A converted to prose, ~220K tokens)
  - LLM-augmented content (~886K tokens)
  - **General English data from FineWeb sample-10BT** (~1.6M tokens) - downloaded via `download_english.py`

- **Scripts**
  - `data/scripts/download_english.py` - Downloads FineWeb sample for 50:50 training balance
  - Updated `data/scripts/count_tokens.py` to include English data directory

- **Documentation**
  - `docs/DOCS.md` - Comprehensive project documentation
  - `docs/RESEARCH.md` - Research reference covering pretraining, post-training, and recent trends

### Decisions Made

#### Model Architecture
- **Framework**: TensorFlow (from existing dependencies and notebook work)
- **Target Size**: 100M parameters (recommended for 8GB VRAM)
- **Architecture Type**: Dense Transformer with RoPE, RMSNorm, SwiGLU
- **Reference Models**: SmolLM2-135M (30 layers, 576 hidden, 9 heads, GQA), TinyStories-28M

#### Tokenizer
- **Choice**: Custom BPE tokenizer with 32K vocabulary (recommended)
- **Rationale**: Domain-optimized, right-sized for 100M parameters, efficient tokenization of ECI-specific terms

#### Training Strategy
- **Data Ratio**: 50% ECI-specific + 50% general English
- **Epochs**: 3 (standard repeat count)
- **Precision**: BF16 mixed precision
- **Optimizer**: AdamW (β1=0.9, β2=0.95, weight decay=0.1)
- **Learning Rate**: 3e-4 with cosine decay schedule
- **Warmup**: 2,000 steps
- **Batch Size**: 1-2 (limited by 8GB VRAM)
- **Gradient Accumulation**: 8 steps (to simulate larger batch)
- **Gradient Checkpointing**: Enabled (~60% activation memory reduction)
- **Sequence Length**: 2,048

#### Post-Training
- **Stage 1**: SFT with LoRA (rank 8-16) on instruct data
- **Stage 2**: DPO (optional) for alignment
- **SFT Data**: `data/instruct/` (56K tokens) + `data/pretrain_expanded/` (220K tokens)

### Research Insights (from DOCS.md & RESEARCH.md)

#### Scaling Laws
- Kaplan (2020): 1.7:1 ratio (underestimated data importance)
- Chinchilla (2022): 20:1 ratio (compute-optimal baseline)
- LLaMA 3 (2024): 1875:1 (over-training with smaller model + more data)
- SmolLM2 (2025): 81,000:1 (extreme over-training at small scale)

#### Key Findings
1. **Data quality > model size** - Phi-4 proved textbook-quality synthetic data beats larger models on noisier data
2. **Synthetic data is essential** - High-quality human data hitting "data wall"
3. **Up to 4 epochs** of repeated data is nearly as good as unique data (JMLR 2025)
4. **SuperBPE (2025)** tokenization reduces tokens by 33%, saves 27% inference compute

#### Post-Training (2025-2026)
1. **RLVR** - Biggest innovation of 2025, verifiable rewards replace expensive human annotation
2. **GRPO** - New standard RL optimizer, no separate critic model needed
3. **DPO** - Practical default for alignment, simpler than RLHF

#### VRAM Guidelines (for 100M params)
| Task | Memory |
|------|--------|
| FP16 Inference | ~200 MB |
| LoRA Training | ~1 GB |
| QLoRA Training | ~0.5 GB |

### Current Data Summary

| Dataset | Tokens | Purpose |
|---------|--------|---------|
| pretrain/ | 495,903 | Original ECI documents |
| pretrain_expanded/ | 220,224 | Q&A expanded to prose |
| pretrain_augmented/ | 886,280 | LLM-augmented content |
| english_pretrain/ | 1,605,297 | General English (FineWeb) |
| **Total Pretrain** | **3,207,704** | **50:50 balanced** |
| instruct/ | 56,375 | SFT/DPO |

### Previous Work (notebooks/old.ipynb)

- Initial TensorFlow implementation of GPT-style language model
- SentencePiece tokenizer training (4K vocab)
- Training loop with Adam optimizer
- Text generation capabilities (free-form and FAQ-style)
- Model config: 4 layers, 4 heads, 128 embedding dim, ~1.3M parameters

### Next Steps (Pending)

1. ~~Train Custom BPE Tokenizer (32K vocab) on combined ECI + English data~~ ✅ DONE
2. **Create binary dataset** (train.bin, val.bin) for training
3. **Build TensorFlow training pipeline** adapted from notebook
4. **Run pretraining** with configured hyperparameters
5. **Post-training** with SFT + optional DPO

### Tokenizer Training (Completed)

- **Model**: `artifact/eci_slm_tokenizer.model`
- **Vocab Size**: 32,000
- **Efficiency**: ~30% fewer tokens than GPT-2 tokenizer on ECI domain text

---

## [0.1.0] - 2026-03-24

### Added
- Project initialization
- ECI domain data collection
- Documentation (DOCS.md, RESEARCH.md)
- Data download scripts
- Token counting utilities

[Unreleased]: https://github.com/anomalyco/eci-slm/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/anomalyco/eci-slm/releases/tag/v0.1.0
