# ECI-SLM Project Documentation

> Last updated: 2026-03-24

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Data Inventory](#2-data-inventory)
3. [Hardware Constraints](#3-hardware-constraints)
4. [Architecture Decisions](#4-architecture-decisions)
5. [Data Requirements & Scaling](#5-data-requirements--scaling)
6. [Training Strategy](#6-training-strategy)
7. [Post-Training Plan](#7-post-training-plan)
8. [Tokenization](#8-tokenization)
9. [Key Takeaways from Research](#9-key-takeaways-from-research)

---

## 1. Project Overview

ECI-SLM is a Small Language Model trained from scratch, focused on the Election Commission of India domain. The model will be trained on a mix of general English and ECI-specific data with a 50:50 ratio.

**Goals:**
- Train a small language model from scratch (not fine-tune an existing one)
- Specialize in Election Commission of India content
- Run on consumer hardware (8GB VRAM)
- Use 50:50 mix of ECI and English data during training

---

## 2. Data Inventory

### Current Data (as of 2026-03-24)

**Pretraining corpus** (`data/pretrain/`) — 11 files, 495,903 GPT-2 tokens:

| File | Tokens (GPT-2) | Words |
|------|----------------|-------|
| eci_overview.txt | 3,251 | 2,591 |
| candidate_handbook_reference.txt | 94,484 | 72,604 |
| conduct_of_elections_guide.txt | 39,464 | 30,686 |
| electoral_roll_blo_guide.txt | 7,927 | 5,813 |
| electoral_process_workflow.txt | 9,276 | 7,103 |
| force_deployment_manual.txt | 50,969 | 37,908 |
| model_code_of_conduct.txt | 5,705 | 4,208 |
| observer_handbook_reference.txt | 24,075 | 18,219 |
| police_handbook_reference.txt | 62,700 | 46,968 |
| returning_officer_sop_manual.txt | 196,099 | 155,598 |
| eci_wikipedia_overview.txt | 1,953 | 1,669 |
| **Total** | **495,903** | **383,367** |

**Pretrain expanded** (`data/pretrain_expanded/`) — 16 files, 220,224 GPT-2 tokens (Q&A converted to prose):

| File | Tokens (GPT-2) | Words |
|------|----------------|-------|
| eci_faq_section_1_q01_expanded.txt | 14,587 | 11,539 |
| eci_faq_section_1_q02_expanded.txt | 25,905 | 20,759 |
| eci_faq_section_1_q03_expanded.txt | 15,330 | 12,427 |
| eci_faq_section_1_q04_expanded.txt | 8,515 | 6,719 |
| eci_faq_section_1_q05_expanded.txt | 9,578 | 7,636 |
| eci_faq_section_1_q06_expanded.txt | 3,855 | 2,870 |
| eci_faq_section_1_q07_expanded.txt | 13,424 | 10,705 |
| eci_faq_section_1_q08_expanded.txt | 689 | 579 |
| eci_faq_section_3_q13_expanded.txt | 3,729 | 2,939 |
| eci_faq_section_3_q14_expanded.txt | 3,481 | 2,757 |
| eci_faq_section_3_q21_expanded.txt | 21,068 | 16,374 |
| eci_faq_section_3_q22_expanded.txt | 1,378 | 1,104 |
| eci_faq_section_3_q23_expanded.txt | 47,930 | 36,523 |
| eci_faq_section_4_q15_expanded.txt | 5,901 | 4,679 |
| eci_faq_section_5_q16_expanded.txt | 33,542 | 27,217 |
| eci_faq_conduct_of_elections_qna_expanded.txt | 11,312 | 9,089 |
| **Total** | **220,224** | **173,916** |

**Pretrain augmented** (`data/pretrain_augmented/`) — 11 files, 886,280 GPT-2 tokens (LLM-augmented prose):

| File | Tokens (GPT-2) | Words |
|------|----------------|-------|
| eci_overview_augmented.txt | 7,722 | 6,446 |
| candidate_handbook_reference_augmented.txt | 151,357 | 132,146 |
| conduct_of_elections_guide_augmented.txt | 83,688 | 71,949 |
| electoral_roll_blo_guide_augmented.txt | 8,056 | 5,813 |
| electoral_process_workflow_augmented.txt | 20,369 | 16,722 |
| force_deployment_manual_augmented.txt | 107,394 | 89,260 |
| model_code_of_conduct_augmented.txt | 5,698 | 4,115 |
| observer_handbook_reference_augmented.txt | 53,326 | 44,672 |
| police_handbook_reference_augmented.txt | 108,662 | 92,807 |
| returning_officer_sop_manual_augmented.txt | 334,042 | 290,963 |
| eci_wikipedia_overview_augmented.txt | 5,966 | 4,999 |
| **Total** | **886,280** | **759,892** |

**Instruct corpus** (`data/instruct/`) — 16 files, 56,375 GPT-2 tokens:

| File | Tokens (GPT-2) | Words |
|------|----------------|-------|
| eci_faq_section_1_q01.txt | 3,836 | 3,006 |
| eci_faq_section_1_q02.txt | 6,272 | 5,096 |
| eci_faq_section_1_q03.txt | 3,552 | 2,795 |
| eci_faq_section_1_q04.txt | 1,954 | 1,434 |
| eci_faq_section_1_q05.txt | 2,240 | 1,783 |
| eci_faq_section_1_q06.txt | 877 | 583 |
| eci_faq_section_1_q07.txt | 3,670 | 2,853 |
| eci_faq_section_1_q08.txt | 185 | 140 |
| eci_faq_section_3_q13.txt | 1,067 | 743 |
| eci_faq_section_3_q14.txt | 992 | 676 |
| eci_faq_section_3_q21.txt | 5,422 | 4,013 |
| eci_faq_section_3_q22.txt | 439 | 355 |
| eci_faq_section_3_q23.txt | 12,882 | 9,788 |
| eci_faq_section_4_q15.txt | 1,721 | 1,344 |
| eci_faq_section_5_q16.txt | 8,518 | 6,511 |
| eci_faq_conduct_of_elections_qna.txt | 2,748 | 1,926 |
| **Total** | **56,375** | **43,046** |

**Grand Total (All ECI):** 54 files, 1,658,782 GPT-2 tokens, 8,811,471 characters.

### Data Summary for Training

| Dataset | Tokens | Purpose |
|---------|--------|---------|
| pretrain/ | 495,903 | Original ECI documents |
| pretrain_expanded/ | 220,224 | Q&A expanded to prose |
| pretrain_augmented/ | 886,280 | LLM-augmented content |
| **Total ECI pretrain** | **1,602,407** | **For 50:50 pretraining** |
| instruct/ | 56,375 | For SFT/DPO |

### Additional Data Needed

| Source | Target Tokens | Purpose | Status |
|--------|--------------|---------|--------|
| English (general) | 1.6M tokens | 50:50 ratio balanced training | To obtain |

---

## 3. Hardware Constraints

| Resource | Available | Limitation |
|----------|-----------|------------|
| Local GPU | 8 GB VRAM | Max ~100M params from scratch |
| Colab GPU | 16 GB VRAM | Max ~500M params from scratch |
| System RAM | 16 GB | Data loading |

### VRAM Budget for 100M Params (FP16)

| Component | Memory |
|-----------|--------|
| Weights | ~200 MB |
| Gradients | ~200 MB |
| Adam optimizer states | ~800 MB |
| Activations (batch=2) | ~1-2 GB |
| **Total** | **~2.5-3.5 GB** |

Fits comfortably on both local 8GB and Colab 16GB.

### Training Memory Formula

```
Total VRAM ≈ (12 bytes × num_params) + activation_memory
```

Where 12 bytes = 2 (weights) + 2 (gradients) + 8 (Adam states: 4 bytes × 2 momentum buffers)

---

## 4. Architecture Decisions

### Architecture (To Be Determined)

Architecture will be determined based on final model size:
- Reference: SmolLM2-135M uses 30 layers, 576 hidden, 9 heads, GQA
- Reference: TinyStories-28M uses 8-16 layers, 256-512 hidden
- Target: Dense Transformer with RoPE, RMSNorm, SwiGLU
- Model size will be 50-150M params depending on data obtained

---

## 5. Data Requirements & Scaling

### Scaling Laws Background

| Law | Year | Ratio | Notes |
|-----|------|-------|-------|
| Kaplan (OpenAI) | 2020 | 1.7:1 | Underestimated data importance |
| Chinchilla (DeepMind) | 2022 | 20:1 | Compute-optimal baseline |
| LLaMA 3 over-training | 2024 | 1875:1 | Smaller model + more data = better inference |
| SmolLM2 | 2025 | 81,000:1 | Extreme over-training at small scale |

### Chinchilla Formula

```
Optimal tokens = 20 × num_parameters
For 100M params: 20 × 100M = 2B tokens
```

### Our Data Plan: 50:50 Balanced Training

| Source | Tokens | Percentage | Purpose |
|--------|--------|-----------|---------|
| ECI pretrain data | 1M-2M | 50% | Domain expertise |
| English (general) | 1M-2M | 50% | General language understanding |
| **Total per epoch** | **2M-4M** | 100% | |
| **Training epochs** | 3 | - | Standard repeat count |

### Why 50:50 Ratio?

- With 1.6M ECI tokens and 3 epochs: 4.8M effective ECI tokens
- Chinchilla-optimal: 4.8M / 20 = 240M params max from ECI alone
- With 1.6M English (50:50): 3.2M total × 3 = 9.6M effective → supports ~160M params
- 50:50 ensures the model doesn't ignore ECI domain (unlike 80:20 English-heavy)

### Data Epochs

Research (JMLR 2025) shows up to 4 epochs of repeated data is nearly as good as unique data:
- 1 epoch = 2-4M tokens
- 2 epochs = 4-8M tokens
- 3 epochs = 6-12M tokens (selected)
- 4 epochs = 8-16M tokens (acceptable)

---

## 6. Training Strategy

### Pretraining Plan

| Stage | Data | Tokens | Purpose |
|-------|------|--------|---------|
| Phase 1 | 50% ECI + 50% English | 2M-4M per epoch | Balanced language + domain |
| Epochs | 3 | - | Standard repeat count |

### Training Hyperparameters

| Parameter | Value | Notes |
|-----------|-------|-------|
| Precision | BF16 mixed precision | Memory efficient, stable |
| Optimizer | AdamW | β1=0.9, β2=0.95, weight decay=0.1 |
| Learning rate | 3e-4 | With cosine decay schedule |
| Warmup | 2,000 steps | Linear warmup |
| Batch size (local) | 1-2 | Limited by 8GB VRAM |
| Batch size (Colab) | 4-8 | Use 16GB VRAM |
| Gradient checkpointing | Enabled | Cuts activation memory ~60% |
| Gradient accumulation | 4-8 steps | Simulate larger batch |
| Sequence length | 2,048 | Standard for small models |

### Model Size (To Be Determined)

Current data supports the following model sizes:
- 1.6M ECI tokens (3 epochs): 4.8M effective → ~240M params max
- 1.6M ECI + 1.6M English (50:50, 3 epochs): 9.6M effective → ~160M params
- Target: **100-150M params** (fits on 8GB VRAM with gradient checkpointing)

### Estimated Training Time

| Setup | Batch Size | Steps | Time Estimate |
|-------|-----------|-------|---------------|
| Local (8GB, 100M) | 2 + grad accum 8 | ~125K | ~10-14 days |
| Colab T4 (16GB) | 4 + grad accum 4 | ~125K | ~5-7 days |
| Colab A100 (16GB) | 8 + grad accum 2 | ~125K | ~2-3 days |

### Compute Estimation

```
Total FLOPs ≈ 6 × params × tokens
           ≈ 6 × 100M × 2B
           ≈ 1.2 × 10^18 FLOPs
```

---

## 7. Post-Training Plan

### Stage 1: Supervised Fine-Tuning (SFT)

| Detail | Value |
|--------|-------|
| Data | `data/instruct/` — 56K GPT-2 tokens, 16 files |
| Additional | `data/pretrain_expanded/` — 220K tokens (Q&A as prose) |
| Format | User/Assistant conversation pairs |
| Method | LoRA (rank 8-16) |
| Epochs | 3-5 |
| Learning rate | 1e-4 |

### Stage 2: DPO (Optional)

| Detail | Value |
|--------|-------|
| Data | Preference pairs from instruct data |
| Method | DPO (no reward model needed) |
| Purpose | Align responses with expected format |

### Method Comparison for Post-Training

| Method | Complexity | Data Needed | Best For |
|--------|-----------|-------------|----------|
| SFT + LoRA | Low | Curated demos | Instruction format |
| DPO | Low | Preference pairs | General alignment |
| GRPO | Medium | Rule-based rewards | Math/code reasoning |
| RLHF (PPO) | Very high | Preference pairs + RM | Maximum control |

**Recommendation for ECI-SLM:** SFT with LoRA first. Add DPO if quality needs improvement.

---

## 8. Tokenization

### Current Setup

- Default: `tiktoken` (GPT-2 tokenizer), vocabulary size 50,257
- Fallback: `sentencepiece` (if `.model` file available)
- Last resort: whitespace splitting

### Tokenizer Options

| Tokenizer | Vocab Size | Pros | Cons |
|-----------|-----------|------|------|
| GPT-2 (tiktoken) | 50,257 | Standard, well-supported | Slightly large vocab for SLM |
| Custom BPE | 32,000 | Right-sized for 100M model | Need to train tokenizer |
| SuperBPE (2025) | Variable | 27% less compute, +4% quality | New, less tooling |

### Recommendation

Train a custom BPE tokenizer with 32K vocab on a mix of general English + ECI data. This gives:
- Vocabulary tuned to your domain
- Right-sized for 100M parameters
- Efficient tokenization of ECI-specific terms

---

## 9. Key Takeaways from Research

### Pretraining

1. **Data quality > model size.** Phi-4 proved a 14B model with textbook-quality synthetic data beats 70B+ models trained on noisier data.
2. **Synthetic data is essential.** High-quality human data is hitting a "data wall." Synthetic generation (paraphrase, expand, generate) multiplies your effective data.
3. **Up to 4 epochs of repeated data** is nearly as good as unique data (JMLR 2025).
4. **SuperBPE (2025)** tokenization reduces tokens by 33% on average and saves 27% inference compute.
5. **DataEvolve (2026)** shows automated data curation strategies outperform manual pipeline design.

### Post-Training

1. **RLVR is the biggest innovation of 2025.** Verifiable rewards (math, code) replace expensive human annotation.
2. **GRPO** is the new standard RL optimizer — no separate critic model needed.
3. **DAPO (2025)** stabilizes long-horizon RL training with dynamic sampling and token-level loss.
4. **DPO is the practical default** for alignment — simpler than RLHF, often matches/exceeds it.
5. **Knowledge distillation** from larger models can transfer reasoning capability (DeepSeek-R1 demonstrated this).
6. **Noisy data is destructive to RLVR.** Even the best algorithms can't compensate for bad annotations.

### Inference-Time Scaling (2025-2026 Trend)

1. Spending more compute during generation (not training) is the new scaling frontier.
2. A 7B model with 100x inference compute can match a 70B model.
3. Inference demand projected to exceed training by 118x by 2026.
4. Reasoning models consume 10-100x more tokens than standard models.
5. Both RLVR scaling and inference-time scaling are predictable; RLHF scaling is not.

### Small Language Models (2026)

1. Market growing: $7.76B (2023) → $20.71B projected by 2030.
2. Sweet spot for enterprise: 1B-7B parameters.
3. 90% cheaper inference than LLM APIs.
4. Privacy advantage: data stays on-device.
5. Leading models: Phi-4 Mini (3.8B), Llama 3.2 3B, Qwen 2.5 3B, SmolLM2 1.7B.

### Architecture Trends

1. **MoE scaled down to SLMs** (Phi-3.5-MoE, Nemotron Nano) for efficiency.
2. **Mamba/SSM hybrids** (Falcon H1R) for linear-time sequence modeling.
3. **MoBA (2025):** Mixture of Block Attention — MoE principles applied to attention.
4. **SWAT (2025):** Sliding window attention with sigmoid + balanced ALiBi + RoPE.

### Continual Learning

1. Predicted as "2026's year of adoption."
2. Core challenge: catastrophic forgetting.
3. Solutions: replay buffers, regularization (EWC), modular LoRA adapters, knowledge distillation.
4. FIT framework (2026): embedding-based filtering + importance-aware selection + targeted layer updates.

### VRAM Rules of Thumb

| Model Size | FP16 Inference | LoRA Training | QLoRA Training |
|-----------|---------------|---------------|----------------|
| 100M | ~200 MB | ~1 GB | ~0.5 GB |
| 500M | ~1 GB | ~4 GB | ~2 GB |
| 1B | ~2 GB | ~8 GB | ~4 GB |
| 3B | ~6 GB | ~20 GB | ~8 GB |
| 7B | ~14 GB | ~40 GB | ~16 GB |

---

## Appendix: Useful Commands

### Count Tokens

```bash
uv run scripts/count_tokens.py
```

### Download FineWeb Subset

```python
from datasets import load_dataset
ds = load_dataset("HuggingFaceFW/fineweb", split="train", streaming=True)
# Stream and save first ~5B tokens
```

### Training Command (Planned)

```bash
# Local (8GB)
uv run scripts/train.py --batch_size 2 --grad_accum 8 --precision bf16

# Colab (16GB)
uv run scripts/train.py --batch_size 4 --grad_accum 4 --precision bf16
```
