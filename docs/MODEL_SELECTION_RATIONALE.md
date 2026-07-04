# Model Selection Rationale — New Sub-1B SLMs

This version intentionally shifts away from older SLMs and prioritizes newer 2025-2026 sub-1B models.

## Primary new models

### LiquidAI/LFM2.5-230M

A very recent 230M model released in June 2026. It is directly relevant because it is described as suitable for edge deployment, agentic tasks, tool use, and data extraction. This should be a top priority.

### LiquidAI/LFM2.5-350M

A recent 350M model in the same family. It provides a useful comparison to the 230M model and may show whether small increases in size improve abstention.

### Qwen/Qwen3.5-0.8B

A newer Qwen sub-1B model. It is near the upper bound of the paper's parameter budget and should act as a strong sub-1B generalist comparison.

### Falcon-H1-Tiny models

Use both the 100M instruction model and the 0.6B reasoning model when possible. This creates a useful contrast between extremely tiny instruction following and tiny reasoning-oriented training.

### IBM Granite 4.0 Nano 350M

Granite 4.0 Nano provides both standard and hybrid variants at 350M. Testing both can reveal whether architecture differences affect schema-abstention.

### FunctionGemma-270M

FunctionGemma is especially relevant because it is function-calling oriented. If it still false-fills unsupported fields, that strengthens the paper's claim. If it performs well, it becomes evidence that task-specific structured-output tuning can reduce schema pressure.

### LFM2-350M-Extract

This extraction-specific model is a strong stress test. It may perform better than general SLMs, but the question is whether it abstains correctly when fields are unsupported.

## Optional backups

- google/gemma-3-270m-it
- Qwen/Qwen3-0.6B

These are recent enough to use as backups, but the main paper should prioritize newer models first.

## Older models to avoid in the main analysis

- SmolLM2-135M/360M
- TinyLlama
- older MobileLLM checkpoints unless specifically needed for a historical baseline

Use them only in a separate "older baseline" appendix.
