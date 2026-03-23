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
| **attention** (self_attn) | ~6 layers | ✅ Best | Most quantization-sensitive; always active, high-information bottleneck |
| **mamba (pre-attn)** | ~6 layers | ✅ Best | Quantization-sensitive; feed into attention — highest impact Mamba layers |
| **mamba (remaining)** | ~40 layers | ⚠️ Low ROI | Quantization-robust — model tolerates weight perturbations here. LoRA budget better spent elsewhere |
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

## 4. Quantization & Layer Sensitivity (§4.2 — Critical for LoRA Targeting)

- FP8 post-training quantization achieves ~99% median accuracy recovery
- KV cache quantized to FP8 enables larger batch sizes

### NVIDIA's Quantization Sensitivity Analysis (Directly Informs LoRA)
NVIDIA performed **quantization sensitivity analysis** across all 52 layers:
- **Self-attention layers (6/52)** are the **most sensitive** components → kept in BF16
- **The 6 Mamba layers that feed into self-attention layers** were also found sensitive → kept in BF16
- The remaining **40 Mamba layers** can be aggressively quantized with minimal accuracy loss
- Conv1D within Mamba layers kept in BF16

> "Keeping the 6 self-attention layers and the 6 Mamba layers in BF16 provided a sweet-spot configuration for accuracy recovery and efficiency trade-off."

### Implication for LoRA
If a layer is sensitive to quantization, it means small weight perturbations there have outsized effects on model quality. This suggests:
- **High-ROI LoRA targets**: The 12 sensitive layers (6 attention + 6 pre-attention Mamba) — perturbations here have the most impact
- **Low-ROI LoRA targets**: The 40 "quantization-safe" Mamba layers — the model is robust to changes in these weights
- **Shared experts**: Always active (2 per layer, all 52 layers) — the paper doesn't call these out as quantization-sensitive, but they're always in the forward pass

### Identifying the 12 Sensitive Layers
The 52 layers follow a repeating pattern (from Figure 2): Mamba layers interleaved with 6 attention layers. The "Mamba layers that feed into self-attention layers" are the Mamba layers immediately preceding each attention layer. To identify them, inspect the model's layer pattern or check which layer indices contain `self_attn` modules.

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
- **RLVR/GRPO**: RL with verifiable rewards could significantly outperform SFT alone (NVIDIA showed RLVR surpassed heavily fine-tuned SFT). We already have deterministic solvers for 4/6 puzzle types that could serve as reward verifiers. See §8 below.
- **DPO**: Even minimal DPO (50 steps, 10k samples) gave meaningful accuracy improvements — could be a cheap add-on after SFT

---

## 8. SFT vs RL & the GRPO/RLVR Path

> Source: [Unsloth × NVIDIA — What are RL environments and how to build them (Mar 2026)](https://unsloth.ai/blog/rl-environments)

### When SFT Falls Short
- **Imitation over adaptivity**: With small datasets, models learn to mimic the answer rather than learn the process to get there.
- **Brittleness**: SFT models struggle when scenarios fall outside training distribution — needs diverse, large datasets.
- RL becomes the better choice as complexity grows: instead of "say exactly this," you provide a goal + verification, so the model explores reasoning paths and becomes resilient to edge cases.

### Recommended Hybrid Strategy
1. **SFT for warm-starting RL** — teach chat template, `\boxed{}` format, general readability. Prevents RL from wasting time learning format.
2. **RL for scaling** — allow model to explore and self-correct. This is where reasoning and robustness are forged.

> "The NVIDIA Nemotron 3 family utilizes SFT as a substantial first stage to ground the model before moving into RL refinement."

### GRPO (Group Relative Policy Optimization)
- Optimized version of PPO: replaces heavy critic/reward models by generating **groups of outputs** and scoring them against a **deterministic verifier**.
- **Reward type**: Typically binary (0 or 1), but supports continuous values. Thrives when environment can programmatically say "Yes" or "No."
- **Efficiency**: Eliminating value model + reward model from PPO significantly reduces memory overhead — key factor in scaling reasoning.
- Available in: **TRL** (already in our stack), **Unsloth**, **NeMo RL**.

### RLVR (RL from Verifiable Rewards)
- Central paradigm shift: replace subjective scoring with **explicit deterministic checks** (correct answer? right tool call?).
- The **environment** becomes the contract between learning and behavior.
- GRPO provides the optimization mechanism; the verifier defines "what good looks like."

### Verification Best Practices
- **Prefer binary rewards**: Strict success/failure signals yield the most stable optimization for GRPO. Partial credit sounds intuitive but hurts training stability.
- **Profile reward signals before training**: Run rollouts across models of varying capabilities. If a frontier model can't consistently outscore the base model, the verifier or task definitions need recalibration.
- **State matching > trajectory matching**: Check final outcome regardless of how agent got there (more robust than comparing against a "golden path").

### Applicability to This Competition
We already have deterministic solvers for 4/6 puzzle types (gravity, unit conversion, base conversion, text encryption). These could serve as binary GRPO verifiers:
- Generate N candidate answers per puzzle → solver checks if answer is correct → reward 0 or 1
- Remaining 2 types (equation transformation, bit manipulation) would need `\boxed{}` string-match against known answer as verifier
- **Blocker**: Competition only accepts LoRA adapter (rank ≤ 32), inference is greedy vLLM — no tool-calling at eval. GRPO would teach the model the reasoning process itself.
- **Compute concern**: GRPO needs many generations per prompt (NVIDIA used 16). Feasible on Kaggle with small batch + gradient accumulation, but slower than pure SFT.

### Relevant Tools & Links
- [NeMo Gym](https://github.com/NVIDIA-NeMo/Gym) — RL environment framework (builds verifiable environments)
- [NeMo RL](https://github.com/NVIDIA-NeMo/RL) — RL training framework
- [NeMo Data Designer](https://github.com/NVIDIA-NeMo/DataDesigner) — synthetic data generation
- [Unsloth GRPO + NeMo Gym tutorial](https://unsloth.ai/docs/models/tutorials/nemotron-3#reinforcement-learning--nemo-gym) — Sudoku + multi-environment training
- [TRL GRPOTrainer](https://huggingface.co/docs/trl/main/en/grpo_trainer) — GRPO in TRL (already in our pip install)

---

## 9. GRPO Research Findings (from link deep-dive)

### Root Cause of reward=0 / loss=0
**`max_completion_length=256` was too short.** Nemotron 3 prepends `<think>` reasoning (token IDs 12/13) before the answer. At temp=0.7, completions are truncated before `\boxed{}` is reached. When all completions in a group get reward=0, advantage=0 for all, so loss=0. Fixed to 512.

### TRL GRPOTrainer — Key Facts
- **Dataset format**: Conversational format (list of message dicts) → `completions` passed to reward func are also lists of message dicts. Our reward function handles this with `isinstance(completion, list)` check.
- **Reward function interface**: Must accept `prompts`, `completions`, `completion_ids`, `trainer_state`, `log_extra`, `log_metric`, plus any dataset column names via `**kwargs`. Must return `list[float]`.
- **Effective batch must be divisible by `num_generations`**: `num_processes * per_device_batch_size * gradient_accumulation_steps` ÷ `num_generations`.
- **`loss_type="dapo"` (default)**: Normalizes by active token count, no length bias. Good for long CoT.
- **`mask_truncated_completions=True`**: Excludes truncated completions from loss. DAPO paper says this is good practice.
- **`scale_rewards`**: Default is `"group"` (std within group). Dr. GRPO paper says `False` avoids difficulty bias. `"batch"` (PPO Lite) is another option.
- **`beta=0.0` (default)**: No KL penalty, no reference model loaded. Saves ~50% memory. Standard practice post-DeepSeek-R1.
- **Logged metrics to watch**: `completions/clipped_ratio` (truncation), `frac_reward_zero_std` (diversity), `reward` (learning signal), `completions/mean_length`.

### NeMo Gym Notebook Reference Settings (Sudoku + Multi-Env)
- `temperature = 1.0` (not 0.7) — more exploration
- `learning_rate = 1e-5` (we use 5e-6 — more conservative, fine for warm-started model)
- `num_generations = 8` (we use 4 — memory constrained)
- `gradient_accumulation_steps = 64` (much higher — they use H100 full finetuning)
- `max_completion_length = max_seq_length - max_prompt_length` (dynamic, not fixed 256)
- `per_device_train_batch_size = 1`
- **100 steps** was enough to see significant improvement (reward 0.15 → 0.6 for sudoku)

### Unsloth RL Blog — Key Insights
- **Binary rewards preferred**: Strict success/failure signals yield most stable optimization for GRPO.
- **SFT for warm-starting RL**: High-quality demonstrations teach format/template, then RL for scaling reasoning.
- **Reward profiling**: Before training, run rollouts across models to confirm reward signal works.
- **Verification logic**: Prefer binary over partial credit. State matching > trajectory matching.

### NVIDIA's Own GRPO Settings (§3.2.5 of paper)
- 128 prompts/step, **16 generations/prompt**, batch 2048
- **Max generation length: 49,152 tokens** (we use 512 — appropriate for our shorter puzzles)
- Router weights frozen during GRPO (already handled in our LoRA config)
- Pure RL on verifiable math/code tasks

### Fixes Applied
1. `GRPO_MAX_COMPLETION`: 256 → 512 (fix truncation before `\boxed{}`)
2. `mask_truncated_completions=True` (exclude truncated completions from loss)
3. Added post-SFT sanity check cell (generates 3 sample completions, checks `\boxed{}` presence)
4. Added `completions/clipped_ratio` and `completions/mean_length` to GRPO logging