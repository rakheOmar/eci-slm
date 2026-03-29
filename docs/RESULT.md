# Results Analysis

This document explains current observed behavior of ECI-SLM, including what is working, what is not, and what to prioritize next.

## Evaluation Setup

The project uses `src/eval.py` with four prompt blocks:

- `general_continuation`
- `eci_continuation`
- `qna_zero_shot`
- `qna_few_shot`

Outputs are saved to:

- `results/results.csv`
- `results/results.json`

## What Looks Good

- Domain vocabulary appears consistently: outputs mention terms like "Returning Officer", "nomination paper", "constituency", and "Election Commission".
- Prompt-type conditioning works at a basic level: QnA prompts produce answer-like continuations rather than random continuation style.
- Few-shot formatting is accepted by the model pipeline and produces coherent structural responses (even when factuality is weak).
- Training pipeline itself is stable: checkpointing, resume/init, best-step tracking, and auto-pruning are in place and functioning.

## What Looks Weak

- Heavy repetition and looping: outputs repeatedly reuse the same clause fragments.
- Weak factual precision: answers are often legally vague or wrong for direct policy questions.
- Limited completion control: responses drift into long, generic bureaucratic text without a clean stop.
- Style over substance: text often sounds official but lacks actionable correctness.
- General prompts also show repetitive, low-information continuation, suggesting limited language depth at current scale/data quality.

## Qualitative Readout By Prompt Type

### General continuation

- Fluency is partial, but semantic density is low.
- Common failure pattern: phrase loops and entity hallucinations.

### ECI continuation

- Strongest category for jargon and domain framing.
- Still suffers from circular sentence structure and duplicated legal fragments.

### QnA zero-shot

- Usually responds in the right style (answer-like text),
- but factual grounding is inconsistent and often incorrect.

### QnA few-shot

- Slight structure improvement versus zero-shot,
- but factual gains are modest; repetition remains the dominant failure mode.

## Why These Failures Happen

- Data imbalance: English background tokens dominate total pool size.
- Small effective context (`block_size=256`) truncates long legal dependencies.
- Augmented corpora include repetitive paraphrase patterns, reinforcing loop-prone text generation.
- Model size is intentionally small (~12.5M params), trading quality for trainability.
- Current evaluation is qualitative; no automatic factual metric yet.

## Practical Interpretation

Current model quality is suitable for:

- prototype experiments,
- architecture/pipeline validation,
- testing data and SFT ideas.

Current model quality is not yet suitable for:

- legal-compliance advice,
- production FAQ automation,
- high-trust election guidance.

## Recommended Next Improvements

1. Improve data quality before scaling size: deduplicate and reduce repetitive augmentation.
2. Increase domain-token share for pretrain runs intended for ECI performance.
3. Expand high-quality instruct examples with strict answer grounding.
4. Add objective eval set (exact-match/keyword-coverage + human rubric) to track factuality.
5. Consider larger context (`block_size` 384 or 512) if compute budget allows.

## Result Confidence

These conclusions are based on the checked-in `results/` outputs and current training/evaluation configuration in `main.py` and `src/eval.py`. Treat them as an iteration snapshot, not final benchmark claims.
