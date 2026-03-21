# Kaggle NVIDIA Nemotron 3 Reasoning Challenge — Notes

> Source: [Nemotron 3 Nano Technical Report (arXiv:2512.20848)](https://arxiv.org/html/2512.20848v1)

---

## 1. Competition Overview

**Goal:** Improve reasoning accuracy on a novel logical-reasoning benchmark using NVIDIA Nemotron-3-Nano-30B via LoRA adapters.

**Submission:** A `submission.zip` containing a LoRA adapter (rank ≤ 32) with `adapter_config.json`.

**Evaluation:** Accuracy. Model generates response via vLLM; answer extracted from `\boxed{}` (fallback: last numeric value). Correct if exact string match OR within relative numerical tolerance.

### Inference Parameters (fixed at eval)
| Parameter | Value |
|---|---|
| max_lora_rank | 32 |
| max_tokens | 7680 |
| top_p | 1.0 |
| temperature | 0.0 (greedy) |
| max_num_seqs | 64 |
| gpu_memory_utilization | 0.85 |
| max_model_len | 8192 |

### Dataset
- **train.csv**: ~9500 logical reasoning puzzles (id, prompt, answer)
- **test.csv**: Several hundred problems (id, prompt) — replaced at scoring time
- **Puzzle types**: Bit manipulation, algebraic equations, number base conversion, gravitational constant, text encryption, unit conversion, equation transformation

---

## 2. Model Architecture (Competition-Relevant Details)

Nemotron 3 Nano 30B-A3B is a **Mixture-of-Experts (MoE) hybrid Mamba-Transformer** model.

### Key Specs
| Property | Value |
|---|---|
| Total params | 31.6B |
| Active params per forward pass | 3.2B (3.6B incl. embeddings) |
| Layers | 52 (6 GQA attention + 46 Mamba-2) |
| Model dimension | 2688 |
| Q-heads / KV-heads / Head dim | 32 / 2 / 128 |
| Mamba state dim / groups / heads / head dim | 128 / 8 / 64 / 64 |
| Expert dimension | 1856 |
| Total routable experts | 128 |
| Active experts per token | 6 |
| Shared experts | 2 |

### Architecture Choices That Matter for LoRA
- **Hybrid Mamba-Transformer**: 6 layers are GQA self-attention, 46 are Mamba-2 SSM layers
- **MoE with granular routing**: 128 routable experts, only 6 active → LoRA on routable experts is wasteful (each expert sees very sparse traffic)
- **Shared experts (2 per layer)**: Always active — good LoRA targets
- **Router**: Learned MLP router with sigmoid gating — **frozen during NVIDIA's RL training** (§3.2.5), suggesting router weights should NOT be LoRA targets
- **No positional embeddings, no dropout, no bias on linear layers**
- **Activation**: Squared ReLU for MoE layers
- **Normalization**: RMSNorm
- **Embeddings**: Untied (embedding ≠ lm_head)

### Implication for LoRA Target Selection
| Layer Group | Count | LoRA Suitability | Rationale |
|---|---|---|---|
| **attention** (self_attn) | ~6 layers | ✅ Good | Always active, high-information bottleneck |
| **mamba** (mixer) | ~46 layers | ✅ Good | Always active, core sequence modeling |
| **shared_expert** | ~2 per layer | ✅ Good | Always active, see all tokens |
| **routable_expert** | 128 per layer | ❌ Poor | Only 6/128 active → sparse gradient signal, huge param count |
| **router/gate** | Per layer | ❌ Avoid | Frozen during NVIDIA's RL; modifying may destabilize routing |
| **embedding/lm_head** | 2 | ⚠️ Risky | Large, may destabilize; untied weights means modifying one doesn't affect other |

---

## 3. Training Recipe Insights (What Worked for NVIDIA)

### Pretraining
- 25T tokens, Warmup-Stable-Decay LR schedule
- Two-phase curriculum: diverse data (94%) then high-quality data (6%)
- Long-context extension: CPT phase with mix of 512k and 4k sequences (121B tokens)
- **Key insight**: Mixing short and long sequences during long-context training preserved short-context benchmarks

### SFT (Supervised Fine-Tuning)
- 18M total samples across: math, code, tool use, long context, formal proofs, multilingual, chat, instruction following, safety, software engineering, science
- **Sequence length**: 256K with packing
- **LR**: 5e-5, 800 warmup steps, 13000 total steps, batch size 64
- **Load balancing**: Sequence-level MoE load balancing, coefficient 1e-4
- **Reasoning control**: 10% of samples stripped of reasoning traces (reasoning on/off), 3% randomly truncated (budget control)
- **Data filtering**: Removed pathological repetition (repeated n-grams), keyword/regex filters for political bias

### RLVR (Reinforcement Learning from Verifiable Rewards)
- **Algorithm**: Synchronous GRPO with masked importance sampling
- **Key settings**: 128 prompts/step, 16 generations/prompt, batch size 2048 (on-policy)
- **Max generation length**: 49K tokens
- **Overlong filtering** boosted reasoning benchmarks
- **Router weights frozen** during RL — preserves routing stability
- **Load balancing**: Aux-loss-free strategy with expert bias updates (update rate 1e-3)
- **Curriculum**: Gaussian sampling that shifts from easy→hard tasks over training. Re-profiled tasks with best checkpoint and constructed new curriculum when progress plateaued.
- **Multi-environment**: Trained ALL environments simultaneously — single-environment training caused unrecoverable degradation of other benchmarks
- **RLVR surpassed heavily fine-tuned SFT** across all evaluated domains

### RLHF
- Used GenRM (generative reward model) instead of Bradley-Terry
- **Group Relative Length Control** to reduce verbose reasoning without hurting accuracy — 30% verbosity reduction
- Circular comparison strategy: O(N) instead of O(N²) GenRM calls

### DPO (Appendix — Not Used in Final Model, but Informative)
- Even 50 steps of DPO with 10k preference samples reduced tool hallucination dramatically
- AIME25: 80.88% → 84.58%, hallucination rate 1.25% → 0%
- GPQA: 65.15% → 69.19%, hallucination rate 8.33% → 0.7%

---

## 4. Quantization Notes

- FP8 post-training quantization achieves ~99% median accuracy recovery
- Self-attention layers (6/52) and preceding Mamba layers most sensitive → kept in BF16
- KV cache quantized to FP8 enables larger batch sizes

---

## 5. Key Benchmark Results (Post-Trained Model)

| Benchmark | Nemotron 3 Nano | Qwen3-30B | GPT-OSS-20B |
|---|---|---|---|
| AIME25 (no tools) | **89.06** | 85.00 | 91.70 |
| AIME25 (with tools) | **99.17** | — | 98.70 |
| GPQA (no tools) | 73.04 | 73.40 | 71.50 |
| LiveCodeBench v6 | **68.25** | 66.00 | 61.00 |
| MMLU-Pro | 78.30 | **80.90** | 75.00 |
| SWE-Bench (OpenHands) | **38.76** | 22.00 | 34.00 |
| IFBench (prompt) | **71.51** | 51.00 | 65.00 |
| RULER-100 @ 1M | **86.34** | 77.50 | — |

---

## 6. Available Resources for Competition

### Open Checkpoints
- [Nemotron 3 Nano 30B-A3B BF16](https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16) — post-trained model
- [Nemotron 3 Nano 30B-A3B Base BF16](https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-Base-BF16) — pretrained base
- [Nemotron 3 Nano 30B-A3B FP8](https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-FP8) — quantized

### Open Datasets
- [Nemotron-CC-v2.1](https://huggingface.co/datasets/nvidia/Nemotron-CC-v2.1) — 2.5T English tokens from Common Crawl
- [Nemotron-CC-Code-v1](https://huggingface.co/datasets/nvidia/Nemotron-CC-Code-v1) — 428B code tokens
- [Nemotron-Pretraining-Code-v2](https://huggingface.co/datasets/nvidia/Nemotron-Pretraining-Code-v2) — curated GitHub + synthetic code
- [Nemotron-Pretraining-Specialized-v1](https://huggingface.co/datasets/nvidia/Nemotron-Pretraining-Specialized-v1) — synthetic STEM reasoning, scientific coding
- [Nemotron SFT Data](https://huggingface.co/collections/nvidia/nemotron-post-training-v3) — SFT datasets
- [Nemotron RL Data](https://huggingface.co/collections/nvidia/nemo-gym) — RL datasets
- [GenRM](https://huggingface.co/nvidia/Qwen3-Nemotron-235B-A22B-GenRM) — generative reward model

### Open Tools
- [NeMo Gym](https://github.com/NVIDIA-NeMo/Gym) — RL environment framework
- [NeMo RL](https://github.com/NVIDIA-NeMo/RL) — RL training framework
- [NeMo Data Designer](https://github.com/NVIDIA-NeMo/DataDesigner) — synthetic data generation

---

## 7. Strategy Notes for This Competition

### Current Approach (notebook)
- Optuna search over layer groups (attention, mamba, shared_expert) and LoRA rank {8, 16, 32}
- SFT with puzzle-type-specific CoT templates
- System prompt + `\boxed{}` formatting
- 400 subsample, 15% val split for search; full data for final training

### Key Considerations
1. **LoRA rank ≤ 32** is enforced at eval. Using rank=32 maximizes capacity.
2. **Greedy decoding** (temp=0) at eval — model must be confidently correct, no sampling tricks.
3. **max_tokens=7680** with max_model_len=8192 — prompts must be compact enough to leave room for generation.
4. **Answer extraction**: `\boxed{}` is prioritized, then heuristic patterns, then last numeric value. SFT must teach the model to reliably use `\boxed{}`.
5. **Puzzle diversity**: Training data has multiple puzzle types — CoT templates per type may help the model recognize and apply domain-specific strategies.

### Ideas to Explore
- **Data augmentation**: Generate more puzzles with known rules to increase training coverage
- **Synthetic reasoning traces**: Use a stronger model to generate step-by-step solutions for training data (NVIDIA used GPT-OSS-120B, DeepSeek-R1 as teacher models)
- **Longer training**: NVIDIA's SFT used 13000 steps with 256K sequence packing — our current setup is much shorter
- **Layer selection refinement**: NVIDIA froze routers and excluded routable experts — our Optuna search confirms this approach
- **Learning rate schedule**: Cosine with warmup (currently used) matches NVIDIA's recipe
- **Prompt sensitivity**: NVIDIA found Nemotron 3 Nano has low prompt sensitivity (std < 1 across benchmarks) — prompt engineering may have diminishing returns vs. data quality
- **RLVR**: If compute allows, RL with verifiable rewards on puzzle tasks could significantly outperform SFT alone (NVIDIA showed RLVR surpassed heavily fine-tuned SFT)
- **DPO**: Even minimal DPO (50 steps, 10k samples) gave meaningful accuracy improvements — could be a cheap add-on after SFT