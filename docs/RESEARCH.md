# Pretraining, Post-Training & Recent Trends in Language Models

> A deep-dive research reference for the ECI-SLM project. Last updated: March 2026.

---

## Table of Contents

1. [Pretraining](#1-pretraining)
   - 1.1 [Objective & Architecture](#11-objective--architecture)
   - 1.2 [Data Curation & Tokenization](#12-data-curation--tokenization)
   - 1.3 [Synthetic Data in Pretraining](#13-synthetic-data-in-pretraining)
   - 1.4 [Scaling Laws & Compute](#14-scaling-laws--compute)
   - 1.5 [Mid-Training](#15-mid-training)
2. [Post-Training](#2-post-training)
   - 2.1 [Supervised Fine-Tuning (SFT)](#21-supervised-fine-tuning-sft)
   - 2.2 [RLHF & Preference Optimization](#22-rlhf--preference-optimization)
   - 2.3 [DPO & Variants](#23-dpo--variants)
   - 2.4 [Reinforcement Learning with Verifiable Rewards (RLVR)](#24-reinforcement-learning-with-verifiable-rewards-rlvr)
   - 2.5 [GRPO & DAPO](#25-grpo--dapo)
   - 2.6 [Knowledge Distillation](#26-knowledge-distillation)
   - 2.7 [Parameter-Efficient Fine-Tuning (PEFT)](#27-parameter-efficient-fine-tuning-peft)
3. [Recent Trends (2024-2026)](#3-recent-trends-2024-2026)
   - 3.1 [Inference-Time Scaling](#31-inference-time-scaling)
   - 3.2 [Reasoning Models](#32-reasoning-models)
   - 3.3 [Small Language Models (SLMs)](#33-small-language-models-slms)
   - 3.4 [Architecture Innovations](#34-architecture-innovations)
   - 3.5 [Continual Learning](#35-continual-learning)
   - 3.6 [Unified Training Pipelines](#36-unified-training-pipelines)
4. [Key Benchmarks & Comparisons](#4-key-benchmarks--comparisons)
5. [Recommendations for ECI-SLM](#5-recommendations-for-eci-slm)
6. [References](#6-references)

---

## 1. Pretraining

### 1.1 Objective & Architecture

**Core Objective:** Pretraining teaches a model general language understanding by predicting the next token across trillions of tokens (10T-20T+ for frontier models, ~300B-3T for SLMs).

**Dominant Architecture:** Decoder-only Transformers with:
- Multi-head or grouped-query attention (GQA)
- Rotary Position Embeddings (RoPE)
- RMSNorm (pre-norm residual connections)
- SwiGLU / SiLU activation in FFN layers

**Training Objectives:**
| Objective | Description | Example Models |
|-----------|-------------|----------------|
| Causal Language Modeling (CLM) | Next-token prediction | GPT series, LLaMA, Phi |
| Prefix LM | Bidirectional on prefix, causal on suffix | PaLM, U-PaLM |
| Masked Language Modeling | Predict masked tokens | BERT, DeBERTa |
| Seq2Seq / Text-to-Text | Encoder-decoder, unified framework | T5, BART, FLAN-T5 |

> **Key Insight:** Next-token prediction remains dominant for both LLMs and SLMs. Data quality matters more than model size—Microsoft proved this with Phi-3/Phi-4, where "textbook-quality" synthetic data on a 14B model outperformed 70B+ models trained on noisier data.

### 1.2 Data Curation & Tokenization

**Data Sources:**
- **Common Crawl**: Backbone of most pretraining corpora, but requires aggressive multi-stage filtering (spam, duplicates, toxic content removal)
- **High-quality curated sources**: Academic literature, Wikipedia, books, code repositories (GitHub)
- **Domain-specific**: STEM references, synthetic math, curated educational content

**Major Datasets:**
| Dataset | Size | Description |
|---------|------|-------------|
| FineWeb (2024-2025) | 15T+ tokens | High-quality filtered English from Common Crawl |
| FineWeb2 | Multi-language | Language-adaptive pipeline for non-English corpora |
| Nemotron-CC | 6.3T tokens | NVIDIA's classifier-ensemble + synthetic rephrasing pipeline |
| DCLM | Large-scale | DataComp for Language Models, quality-focused filtering |
| Nemotron-CLIMB | 1.2T tokens | Cluster-based iterative data mixture optimization |

**Curation Pipeline (State-of-the-Art 2025-2026):**
1. **HTML extraction** (jusText, resiliparse)
2. **Language identification** (FastText, CLD3)
3. **Heuristic filtering** (length, language detection, perplexity thresholds)
4. **Quality classifiers** (FastText-based, fine-tuned LLM classifiers)
5. **Semantic deduplication** (MinHash LSH, SimHash, embedding-based)
6. **Synthetic data rephrasing** (LLM-based rewriting of low-quality documents)
7. **Token-level filtering** (2025-2026 trend: masking/replacing low-value tokens during training)

**DataEvolve (2026):** A framework that evolves data curation strategies through iterative optimization rather than manual design. Evolved strategies converge on cleaning-focused approaches: targeted noise removal and format normalization with domain-aware preservation.

**Tokenization Innovations:**
- **BPE (Byte-Pair Encoding):** Standard for most models; vocabulary sizes 32K-256K
- **SuperBPE (2025):** Extends BPE beyond subwords to "superwords" that bridge whitespace boundaries. 33% fewer tokens on average, +4.0% on downstream tasks, 27% less inference compute
- **QA-Token (2026):** Quality-aware tokenization that incorporates data reliability into vocabulary construction via bilevel optimization

**Multilingual Data Curation:**
- **JQL (2025):** Distills LLM annotation capabilities into lightweight classifiers for efficient multilingual data filtering
- **FineWeb2 Pipeline:** Automatically adapts filtering/deduplication to any language using language-specific statistics

### 1.3 Synthetic Data in Pretraining

Synthetic data has become the most significant trend in pretraining data strategy (2024-2026).

**Why Synthetic Data?**
- High-quality human-generated data is hitting a "data wall" (estimated ~300-500T tokens of public text, much low-quality)
- Synthetic data can be structured for "progression-oriented" learning (step-by-step logic)
- Synthetic data yields 5-10x faster convergence to the same loss

**Phi Model Series (Microsoft):**
| Model | Params | Data Strategy | Key Finding |
|-------|--------|---------------|-------------|
| Phi-1.5 (2023) | 1.3B | 30B synthetic tokens | "Textbook-quality" data beats 10x more web data |
| Phi-2 (2023) | 2.7B | Synthetic + filtered web | Competitive with 25x larger models |
| Phi-3 (2024) | 3.8B | Distilled from GPT-4 | 90%+ capability at 5% size |
| Phi-4 (2024-2025) | 14B | 400B unique synthetic tokens, 50 dataset types | Surpasses teacher on STEM benchmarks |

**Phi-4 Data Generation Techniques:**
- Multi-agent prompting (collaborative AI debate)
- Self-revision workflows (generate → critique → rewrite)
- Instruction reversal (derive prompts from complex information)
- Synthetic data constitutes the "bulk" of training data; organic data provides "what" (facts), synthetic provides "how" (reasoning)

**Synthetic Data Risks:**
- Pure synthetic training risks model collapse (distributional convergence)
- Higher hallucination rates without organic data grounding
- Best results: anchor synthetic data in human-verified "truth" content

**Key Frameworks:**
- **Synthetic Bootstrapped Pretraining (SBP, 2025):** Models inter-document relations, synthesizes new corpus for joint training. Up to 60% of oracle improvement.
- **ReWire (2025):** Recycles low-quality documents through guided LLM rewriting instead of discarding them.
- **Nemotron-Synth:** NVIDIA's reusable synthetic dataset pipeline integrated into NeMo Curator.

### 1.4 Scaling Laws & Compute

**Three Axes of Scaling (as of 2026):**

1. **Pre-training Scaling:** Power-law relationship between compute/data and performance. Still fundamental but "low-hanging fruit" mostly exhausted.
2. **Reinforcement Learning Scaling (RLVR):** Verifiable rewards produce predictable scaling. Does not apply to RLHF.
3. **Inference-Time Scaling:** Allocating more compute during generation produces predictable improvement on complex tasks.

**Compute-Efficient Training for SLMs:**
- LoRA/QLoRA: 10-20x compute reduction for fine-tuning, 80-95% of full fine-tuning quality
- Mixed-precision training (BF16/FP16 + FP32 master weights)
- Gradient checkpointing for memory optimization
- Flash Attention: Near-linear memory complexity, 2-4x speedup

**Cost Benchmarks (2026):**
| Path | Compute Cost | Timeline | Data Needed |
|------|-------------|----------|-------------|
| Train from scratch (sub-1B) | $500-$5,000 | Weeks-months | Millions of samples |
| Fine-tune pre-trained (LoRA) | $10-$100/run | Hours-days | Hundreds-thousands |
| Knowledge distillation | $50-$500 | Days-weeks | Teacher-generated |

### 1.5 Mid-Training

A bridge phase between regular pre-training and post-training, gaining prominence since 2024.

**Techniques grouped under "mid-training":**
- Long-context extension (extending context from 4K → 128K+)
- Domain-specific data mixing
- Synthetic data augmentation
- Curriculum-based data ordering
- Tokenizer refinement or re-tokenization

> **Key Insight from Sebastian Raschka:** "2024 saw all major labs make their pre-training pipelines more sophisticated by focusing on synthetic data, optimizing data mixes, using domain-specific data, and adding dedicated long-context training stages."

---

## 2. Post-Training

Post-training transforms a raw pre-trained model into a useful, aligned, instruction-following system. As of 2026, post-training accounts for the majority of a model's usable capability.

### 2.1 Supervised Fine-Tuning (SFT)

The canonical first step of post-training: teaching the model instruction-following format.

**Process:**
- Fine-tune on 1-10M curated instruction-response pairs
- Typically uses a curated subset of diverse tasks
- Nemotron 3 Super used 7M samples from a broader corpus of 40M

**SFT Data Quality Matters Most:**
- Curated demonstrations outperform raw web data
- High-quality examples from capable models (or humans) are critical
- Task diversity prevents overfitting to specific patterns

**Techniques:**
| Technique | Description | VRAM Requirement |
|-----------|-------------|-----------------|
| Full Fine-Tuning | Update all parameters | High (model + gradients) |
| LoRA | Freeze base, train low-rank adapters | ~25-50% of full |
| QLoRA | 4-bit quantized base + LoRA | ~10-25% of full |
| Prompt Tuning | Learn soft prompt embeddings | Minimal |
| Prefix Tuning | Learn virtual prefix tokens | Low |

### 2.2 RLHF & Preference Optimization

**RLHF Pipeline (Classic - 2022):**
1. **SFT:** Fine-tune on demonstrations
2. **Reward Model Training:** Train a separate model on human preference pairs
3. **RL Fine-Tuning (PPO):** Optimize LLM policy against reward model

**Why RLHF Is Complex:**
- Requires training/loading 4 models simultaneously (policy, reference, reward model, value function)
- PPO is sensitive to hyperparameters
- Reward hacking risk (model exploits reward without improving quality)
- Expensive human annotation

**Key Finding:** InstructGPT showed that a 1.3B RLHF model outperformed 175B GPT-3—a landmark demonstration that alignment matters more than scale.

### 2.3 DPO & Variants

**Direct Preference Optimization (DPO, 2023):**
- Eliminates the reward model entirely
- Frames alignment as a classification problem on preference pairs
- Directly optimizes the policy from (chosen, rejected) response pairs
- Stable, simple, fast to implement

**DPO vs RLHF Comparison:**
| Feature | RLHF (PPO) | DPO |
|---------|-----------|-----|
| Complexity | Very high (4 models) | Low (2 models) |
| Stability | Finicky | Stable (supervised loss) |
| Compute | Very expensive | Significantly cheaper |
| Performance | Strong | Often matches/exceeds PPO |
| Reward hacking risk | Yes | Minimal |

**DPO Variants (2024-2026):**
| Method | Year | Key Innovation | Data Requirement |
|--------|------|---------------|-----------------|
| DPO | 2023 | Implicit reward, no RM needed | Preference pairs |
| IPO | 2024 | No reliance on Bradley-Terry assumptions | Preference pairs |
| KTO | 2024 | Prospect theory, no pairing needed | Binary labels |
| SimPO | 2024 | Simplified preference optimization | Preference pairs |
| ORPO | 2024 | Merges SFT + preference optimization | Single training phase |
| Iterative DPO | 2024-2025 | Online generation + re-labeling | Dynamic preference pairs |
| SPIN | 2024 | Self-play fine-tuning | Model's own outputs |

> **Practitioner Recommendation:** Start with DPO. It is currently the most streamlined and cost-effective alignment method. Scale to full RLHF or GRPO as quality requirements grow.

### 2.4 Reinforcement Learning with Verifiable Rewards (RLVR)

**The biggest post-training innovation of 2025.**

**Core Idea:** For tasks where correctness can be automatically verified (math, code, structured reasoning), use programmatic verification instead of human preference labels.

**Why RLVR Matters:**
- Eliminates expensive human annotation bottleneck
- Faster, cheaper, more consistent than human judgment
- Can scale to millions of training examples
- Produces emergent reasoning capabilities (self-reflection, dynamic strategy adaptation)

**DeepSeek-R1 Breakthrough (Jan 2025):**
- Demonstrated that pure RL with verifiable rewards can produce reasoning models without any human-labeled reasoning traces
- Matched OpenAI o1 on multiple benchmarks
- Proved that reasoning can emerge from RL alone (R1-Zero approach)

**RLVR Pipeline:**
1. Start with a base model (typically already SFT'd)
2. Provide questions with verifiable answers (math solutions, code that passes tests)
3. Sample multiple completions per prompt
4. Verify correctness programmatically
5. Update policy using RL (GRPO/DAPO/PPO)

**Domains Where RLVR Works:**
- Mathematics (proof checkers, symbolic verification)
- Code execution (unit tests, compiler checks)
- Formal logic (proof assistants)
- Expanding: chemistry, biology, game environments (2026+)

### 2.5 GRPO & DAPO

**GRPO (Group Relative Policy Optimization, DeepSeek 2024):**
- Samples multiple responses per prompt
- Computes advantages by comparing within the group (no separate critic model)
- Eliminates value network, reducing memory/compute
- Asymptotically efficient (U-statistic property)
- Now the standard RL optimizer for reasoning models

**How GRPO Works:**
1. For each prompt, sample G completions
2. Compute rewards for each completion
3. Calculate group-relative advantages (normalize within group)
4. Apply clipped policy gradient with KL penalty

**DAPO (Dynamic sAmpling Policy Optimization, ByteDance/Tsinghua 2025):**
Stabilizes long-horizon RL training through:
- **Clip-Higher:** Allows larger policy updates without collapse
- **Dynamic Sampling:** Resamples failed groups to reduce gradient noise
- **Token-Level Loss:** Applies policy gradient at token level (not just sequence)
- **No KL Penalty:** Removes conservative KL constraint
- **Zero-Gradient Filtering:** Discards uninformative gradient signals

**DAPO Results:** On AIME 2024, DAPO trained Qwen2.5-32B to 50 points, outperforming DeepSeek-R1-Zero with 50% fewer training steps.

**Other GRPO Variants (2025):**
- **Dr. GRPO:** Bias mitigation for gradient estimation
- **TIS:** Truncated importance sampling + dynamic sampling
- **SAPO:** Adaptive clipping for stable training
- **RISE:** Joint problem-solving + self-verification training

**Important Finding (2026):** Noisy data is destructive to RLVR. Even the best algorithmic improvements (DAPO, SAPO, Dr. GRPO) fail to compensate for noisy annotations. Training on 50% incorrect annotations underperforms clean data by ~8-10%. Data quality remains paramount.

### 2.6 Knowledge Distillation

Transferring knowledge from a large "teacher" model to a compact "student" model.

**Distillation Types:**
| Type | What is Transferred | Method |
|------|-------------------|--------|
| Response-based | Output probabilities (soft targets) | Temperature-scaled softmax matching |
| Feature-based | Intermediate representations | Hidden state alignment |
| Relation-based | Relationships between samples | Similarity/distillation loss |

**Modern Distillation Approaches (2025-2026):**
| Approach | Key Innovation |
|----------|---------------|
| **DeepSeek-R1 Distillation** | Distilled reasoning capability; 94.5 on MATH-500 at 70B. Distilled small models outperform RL-trained small models |
| **Multi-Step KD (MSKD)** | Staged transfer (teacher → mid → student) bridges capacity gap better than direct 1-step |
| **Concrete Score Distillation (CSD)** | Discrete score-matching objective; avoids softmax smoothing; flexible logit alignment |
| **Self-Distillation** | Model distills its own predictions over epochs |
| **LoRA Distillation** | Transfers adapter weights into student base model |
| **Cross-Modal Distillation** | Transfer from multimodal teacher to lightweight encoders |

**Compression Sequence:** Pruning → Knowledge Distillation → Quantization (P-KD-Q) achieves maximum compression while preserving quality. Order matters: pruning removes redundancy first, distillation recovers lost knowledge, quantization compresses final architecture.

**Economics of Distillation:**
- 5-30x cost reduction vs training from scratch
- 4x faster inference
- 95-97% of original performance maintained
- Can achieve <3% of original training data for effective transfer

### 2.7 Parameter-Efficient Fine-Tuning (PEFT)

**The dominant paradigm for adapting models, especially SLMs.**

**LoRA (Low-Rank Adaptation):**
- Freezes base model, trains small low-rank matrices
- Typical rank (r): 8-16 for most tasks
- Memory: ~25-50% of full fine-tuning
- Can run on consumer GPUs (RTX 3060 6GB for 3B models)

**QLoRA:** Combines 4-bit quantization (NF4) with LoRA
- Fine-tune 7B model on RTX 4090
- 80-90% of full fine-tuning performance
- 39% runtime increase vs LoRA

**Decision Framework:**
| Scenario | Recommendation |
|----------|---------------|
| Multi-adapter serving planned | LoRA |
| Fast iteration, 90-95% quality | LoRA |
| 80-90% acceptable, large model | QLoRA |
| Substantial compute available | Full fine-tuning |
| First-time alignment | DPO |

---

## 3. Recent Trends (2024-2026)

### 3.1 Inference-Time Scaling

**The paradigm shift of 2025-2026.** Instead of training bigger models, spend more compute during generation.

**Core Insight:** A 7B parameter model with 100x inference compute can match a 70B model with standard inference.

**Techniques:**
| Technique | Description |
|-----------|-------------|
| **Chain-of-Thought (CoT)** | Prompting model to show reasoning steps |
| **Self-Consistency** | Sample multiple reasoning paths, majority vote |
| **Best-of-N Ranking** | Generate N candidates, select best by verifier |
| **Rejection Sampling** | Generate until verifier approves |
| **Self-Refinement** | Generate → critique → rewrite loop |
| **Search Over Solution Paths** | Tree/graph search over reasoning steps |
| **Process Reward Models** | Feedback on each reasoning step, not just final answer |
| **Parallel Reasoning (ThreadWeaver)** | 1.53x speedup with adaptive threading |

**Infrastructure Impact (2026):**
- Inference demand projected to exceed training by 118x
- By 2030, inference could claim 75% of total AI compute ($7T infrastructure investment)
- Inference spending surpassed training spending in early 2026 for the first time
- Reasoning models consume 10-100x more tokens than standard models

**Scaling Laws:**
- For every 10x increase in inference-time compute, predictable increase in reasoning benchmark performance
- Both RLVR scaling and inference-time scaling are predictable; RLHF scaling is not

### 3.2 Reasoning Models

**The dominant theme of 2025 LLM development.**

**Timeline of Focus:**
| Year | Focus |
|------|-------|
| 2022 | RLHF + PPO (ChatGPT) |
| 2023 | LoRA SFT |
| 2024 | Mid-Training (synthetic data, data mixing) |
| 2025 | RLVR + GRPO (reasoning models) |
| 2026 (predicted) | RLVR extensions, more inference-time scaling |
| 2027 (predicted) | Continual learning |

**Key Reasoning Models:**
- **OpenAI o1/o3:** First "thinking" models, extended CoT
- **DeepSeek-R1:** Open-weight reasoning model, pure RL with verifiable rewards
- **QwQ-32B:** Alibaba's reasoning model using GRPO
- **Kimi k1.5:** Reasoning via long-context RL
- **MiMo:** Reasoning unlocked from pretraining to posttraining

**What Makes Reasoning Models Different:**
- Generate long chain-of-thought (thousands of tokens) before final answer
- Self-reflect and backtrack during reasoning
- Use "thinking" tokens not shown to users
- Dynamically adjust strategy based on problem complexity

### 3.3 Small Language Models (SLMs)

**Market Growth:** $7.76B (2023) → projected $20.71B by 2030.

**Definition:** Models with fewer than 10-14B parameters. Sweet spot for enterprise: 1B-7B.

**Leading SLMs (2026):**
| Model | Params | Architecture | Key Feature |
|-------|--------|-------------|-------------|
| Phi-4 Mini | 3.8B | Dense | Best reasoning at 3-4B range |
| Llama 3.2 3B | 3B | Dense | Strong generalist |
| Qwen 2.5 3B | 3B | Dense | 20+ languages |
| Gemma 3 | 4B | Dense | Multilingual, Google ecosystem |
| SmolLM2 | 1.7B | Dense | Trained on FineMath, Stack-Edu |
| DeepSeek-R1-Distill | 1.5B | Distilled | Reasoning capability transfer |
| Phi-3.5-MoE | 41.9B total, 6.6B active | MoE | High performance with sparse activation |
| Falcon H1R 7B | 7B | Hybrid Transformer-Mamba | Hybrid architecture |
| Nemotron Nano 30B | 31.6B total, 3.6B active | MoE + Mamba-2 | 1M context window |
| Apriel 1.6 15B | 15B | Multimodal | Competitive with 235B models |

**Why SLMs Matter:**
- 90% cheaper inference than LLM APIs
- Run on consumer hardware, edge devices, smartphones
- Faster inference (sub-second latency)
- Privacy: data stays on-device
- Easier to fine-tune for domain-specific tasks

### 3.4 Architecture Innovations

**Mixture-of-Experts (MoE):**
- Router selects subset of expert FFN layers per token (e.g., 2 of 16)
- High total capacity with low active parameters
- Scaled down to SLMs: Phi-3.5-MoE, Nemotron Nano
- **MoBA (2025):** Mixture of Block Attention—applies MoE principles to attention mechanism for long-context efficiency

**State Space Models (SSMs) & Hybrids:**
- **Mamba / Mamba-2:** Linear-time sequence modeling alternative to quadratic attention
- **Hybrid Transformer-Mamba:** Falcon H1R combines both for efficiency
- **MossNet (2025):** Mixture of State-Space Experts emulating multi-head attention; outperforms pure Transformer and SSM baselines

**Attention Innovations:**
| Innovation | Description |
|-----------|-------------|
| Grouped-Query Attention (GQA) | Share KV heads across query heads |
| Sliding Window Attention | Linear complexity per token |
| SWAT (2025) | Sigmoid + balanced ALiBi + RoPE for efficient SWA |
| Flash Attention 2/3 | Hardware-aware exact attention |

**Tokenization:**
- **SuperBPE (2025):** Superword tokenization bridging whitespace; +4% on 30 tasks, 27% less compute

### 3.5 Continual Learning

**Predicted as "2026's year of adoption" by DeepMind researchers.**

**Core Challenge:** Catastrophic forgetting—training on new data erases old capabilities.

**Three Drivers of Forgetting:**
1. Cumulative redundancy from semantically similar requests
2. Unstable gradient updates
3. Excessive parameter drift

**Approaches:**
| Approach | Mechanism |
|----------|-----------|
| **Replay (Rehearsal)** | Maintain buffer of old examples mixed into new training |
| **Regularization (EWC, etc.)** | Penalize changes to important parameters |
| **Modular Adapters** | Separate LoRA adapters per domain; merge/retire over time |
| **Knowledge Distillation** | Ensure new model's representations remain consistent with old |
| **FIT Framework (2026)** | Embedding-based filtering + importance-aware adaptive selection + targeted layer updates |

**Continual Learning Stages for LLMs:**
1. Continual pre-training (acquiring new knowledge)
2. Continual fine-tuning (adapting to new tasks)
3. Continual alignment (maintaining safety/behavior)

> **Open Challenge:** True continual learning (autonomous weight updates from deployment) vs. periodic offline retraining + RAG. The field has not yet solved seamless knowledge integration at scale.

### 3.6 Unified Training Pipelines

**Trend: Merging training stages into single objectives.**

- **ORPO (2024):** Already merges SFT and preference optimization
- **Future Direction:** Single objective handling instruction following + preference alignment + reasoning
- **Environment-native training:** Shift from static datasets to interactive environments (NeMo Gym, RLFactory) for multi-step tool use

**Modern Post-Training Pipeline (2026):**
```
Pre-training → Mid-Training → SFT → Preference Optimization → RL (Verifiable Rewards)
```
Each stage solves a different problem:
- **SFT:** Format (instruction following, structured output)
- **Preference:** Values (human alignment, safety)
- **RL:** Capability (reasoning, discovery of new strategies)

---

## 4. Key Benchmarks & Comparisons

### Post-Training Method Comparison (2026)

| Method | Complexity | Data Needed | Best For |
|--------|-----------|-------------|----------|
| SFT | Low | Curated demonstrations | Instruction format |
| DPO | Low | Preference pairs | General alignment |
| GRPO | Medium | Rule-based rewards | Math/code reasoning |
| RLHF (PPO) | Very high | Preference pairs + RM | Maximum control |
| RLVR | Medium | Verifiable tasks | Reasoning capability |
| Knowledge Distillation | Medium | Teacher outputs | Model compression |

### Evolution of Alignment

```
RLHF (2022) → DPO (2023) → Iterative DPO (2024) → RLVR/GRPO (2025) → Unified pipelines (2026+)
```

---

## 5. Recommendations for ECI-SLM

Based on this research, the following strategy is recommended for the ECI-SLM project:

### Pretraining Strategy
1. **Base Architecture:** Dense Transformer with GQA, RoPE, RMSNorm, SwiGLU (proven at 1-7B scale)
2. **Data:** Curate high-quality domain-specific data; supplement with textbook-quality synthetic data
3. **Tokenization:** Consider SuperBPE or enhanced BPE for efficiency gains
4. **Data Pipeline:** Implement multi-stage filtering (heuristic → classifier → dedup → synthetic enrichment)

### Post-Training Strategy
1. **Stage 1 - SFT:** Curate 1K-10K high-quality instruction examples; fine-tune with LoRA
2. **Stage 2 - DPO:** Collect preference pairs for alignment; simpler and more stable than RLHF
3. **Stage 3 - GRPO (if applicable):** For math/code/reasoning tasks with verifiable outputs
4. **Distillation:** Use a larger teacher model to generate training data for the SLM

### Practical Considerations
- **LoRA rank 8-16** for most tasks
- **QLoRA** for consumer GPU deployment (RTX 4090 for 7B models, RTX 3060 for 3B)
- **Data quality > model size** (Phi-4 lesson)
- **Plan for continual learning** from the start (model drift is real)
- **Inference-time scaling** as a complementary strategy to training

---

## 6. References

### Pretraining
- Gunasekar et al. (2023). "Textbooks Are All You Need." Phi-1 paper.
- Li et al. (2023). "Textbooks Are All You Need II." Phi-1.5 paper.
- Abdin et al. (2024). "Phi-3 Technical Report." Microsoft.
- Penedo et al. (2024). "FineWeb: Decanting the Web for the Finest Text Data."
- Su et al. (2025). "Nemotron-CC: Transforming Common Crawl into a Refined Long-Horizon Pretraining Dataset." ACL 2025.
- Diao et al. (2025). "Nemotron-CLIMB: CLustering-based Iterative Data Mixture Bootstrapping." NeurIPS 2025.
- Qin et al. (2026). "DataEvolve: AI can Autonomously Evolve Pretraining Data Curation."
- Liu et al. (2025). "SuperBPE: Space Travel for Language Models."
- Yang et al. (2025). "Synthetic Bootstrapped Pretraining (SBP)."

### Post-Training
- Ouyang et al. (2022). "Training Language Models to Follow Instructions with Human Feedback." InstructGPT/RLHF.
- Rafailov et al. (2023). "Direct Preference Optimization: Your Language Model is Secretly a Reward Model." NeurIPS 2023.
- Shao et al. (2024). "DeepSeekMath: Pushing the Limits of Mathematical Reasoning." (Introduced GRPO)
- DeepSeek-AI (2025). "DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning."
- Yu et al. (2025). "DAPO: an Open-Source LLM Reinforcement Learning System at Scale."
- Wolfe, C.R. (2025-2026). "GRPO++: Tricks for Making RL Actually Work."
- Wolfe, C.R. (2026). "Continual Learning with RL for LLMs."

### Knowledge Distillation
- Hinton et al. (2015). "Distilling the Knowledge in a Neural Network."
- Yim et al. (2026). "Beyond One-Step Distillation: Bridging the Capacity Gap via Multi-Step Knowledge Transfer." EACL 2026.
- CSD (2026). "Distillation of Large Language Models via Concrete Score Matching." ICLR 2026.
- Vanderbilt Survey (2026). "Knowledge distillation and dataset distillation of LLMs: emerging trends."

### Architecture & Efficiency
- Fu et al. (2025). "SWAT: Sliding Window Attention Training for Efficient LLMs."
- Tuli et al. (2025). "MossNet: Mixture of State-Space Experts is a Multi-Head Attention."
- MoBA (2025). "Mixture of Block Attention for Long-Context LLMs." NeurIPS 2025.

### Surveys & Overviews
- Raschka, S. (2025). "The State of LLMs 2025: Progress, Problems, and Predictions."
- LLM Stats (2026). "Post-Training in 2026: GRPO, DAPO, RLVR & Beyond."
- Zylos Research (2026). "LLM Fine-tuning Techniques 2026."
- Chen et al. (2026). "Continual Learning in Large Language Models: Methods, Challenges, and Opportunities." arXiv:2603.12658.
