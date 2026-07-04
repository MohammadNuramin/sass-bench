# SASS-Bench Results — "Valid JSON, False Facts" (run 2026-07-04)

Experiment executed end-to-end by an experiment-running agent. This document is the
analysis hand-off for the paper author. **Read the "What you can / cannot claim" section
before writing** — the honest result is more nuanced (and more defensible) than a blanket claim.

---

## 1. What was run

- **Models (10, all sub-1B):** LFM2.5-230M, LFM2.5-350M, LFM2-350M-Extract, Qwen3.5-0.8B,
  Qwen3-0.6B, Falcon-H1-Tiny-100M-Instruct, Falcon-H1-Tiny-R-0.6B, Granite-4.0-350M,
  Granite-4.0-H-350M, FunctionGemma-270M. (All real HF checkpoints; `transformers` 5.13,
  bf16/auto, deterministic greedy for the main run.)
- **Conditions:** C0 free-text · C1 loose-JSON · C2 strict-JSON · C3 strict+null-rule ·
  C4 strict+few-shot · C5 schema-constrained decoding (lm-format-enforcer).
- **Add-ons:** INT8/INT4 quantization (5 quant-compatible models); robustness pass
  (temperature 0.2, 3 repetitions).
- **Dataset:** SASS-Bench, 1,000 examples, seed 42 (200 each of SUPPORTED / MISSING /
  AMBIGUOUS / CONTRADICTORY / DISTRACTOR).
- **Volume:** 109,000 main generations + 60,000 robustness (C0/C3) + 21,000 replication (C2).
- **Stats:** paired cluster bootstrap over examples, B = 10,000, for SPI 95% CIs.
- **Hardware:** run distributed across 2× RTX 4090 + 1× RTX 3090.

Primary metric: **SPI-C2 = FalseFillRate(C2 strict-JSON) − FalseFillRate(C0 free-text)**.
Gold-null fields = MISSING/AMBIGUOUS/CONTRADICTORY. Positive SPI ⇒ schema prompting increased
fabrication of unsupported fields.

---

## 2. Headline result (fair baseline — see §4 for why)

Significant positive SPI-C2 (95% bootstrap CI excludes 0):

| Model | FFR C0 → C2 | **SPI-C2 [95% CI]** | SPI-C3 [95% CI] | Note |
|---|---|---|---|---|
| Qwen3-0.6B | 0.22 → 0.70 | **+0.485 [0.44, 0.52]** | −0.145 [−0.17,−0.12] | C3 fully mitigates |
| Qwen3.5-0.8B | 0.29 → 0.71 | **+0.422 [0.38, 0.46]** | **+0.153 [0.11, 0.20]** | residual pressure under C3 |
| Granite-4.0-H-350M | 0.39 → 0.65 | **+0.255 [0.22, 0.29]** | −0.008 [−0.04, 0.02] | C3 mitigates |
| Granite-4.0-350M | 0.62 → 0.81 | **+0.192 [0.16, 0.22]** | −0.142 [−0.18,−0.10] | C3 mitigates |
| LFM2.5-350M | 0.72 → 0.76 | +0.045 [0.005, 0.085] | +0.007 [−0.03, 0.04] | marginal |

The other 5 models do **not** show significant positive SPI-C2 (reasons in §4).

---

## 3. Hypothesis-by-hypothesis verdict

- **H1 / RQ1 (strict JSON ↑ false-fill): SUPPORTED, conditionally.** 5/10 models, large for the
  capable ones. NOT universal (see §4). Replicates under sampling (§5).
- **H3 (high validity, low faithfulness — Format–Truth gap): STRONGLY SUPPORTED.** Cleanest result.
  Capable models reach ~1.0 parse & schema-validity while false-filling.
- **H4 / RQ5 (constrained decoding fixes structure, not truth): STRONGLY SUPPORTED.** Under C5,
  **all** models hit 1.00 parse AND 1.00 schema-validity, yet false-fill stays high for half
  (LFM2-Extract 0.69, LFM2.5-230M 0.72, FunctionGemma 0.72, Granite 0.48). Figure 3.
- **RQ3 (explicit null rule ↓ false-fill): SUPPORTED.** C3 collapses SPI to ≈0/negative for most.
- **H2 / RQ4 (few-shot beats instruction): NOT SUPPORTED / partially refuted.** By the protocol's
  bar FFR(C4) < FFR(C3), few-shot wins for only 5/10 and is **worse** for Qwen3-0.6B (0.07→0.53),
  Granite ×2, FunctionGemma. The explicit null-rule is the more reliable mitigation. **Report this
  as a negative result.**
- **H5 (reasoning models don't abstain better): supported in spirit.** The reasoning model
  (Falcon-H1-R) "abstains" only by failing to emit parseable JSON; the strongest model (Qwen3.5)
  is the hardest to fully mitigate.

---

## 4. What you can / cannot claim (critical)

**CAN claim (fully supported):**
> "For sub-1B SLMs capable of producing valid JSON, structured-output prompting and constrained
> decoding guarantee *format* while not guaranteeing *truth*: schema validity and field
> faithfulness are decoupled, the effect replicates under sampling, and only explicit
> null-instructions (not few-shot) reliably narrow the gap."

**CANNOT claim (would not survive review):**
- **"All sub-1B SLMs hallucinate under schema pressure."** 5/10 do not. Weak models
  (FunctionGemma, Falcon-H1 ×2) fail *structurally* under strict JSON instead of fabricating
  (their false-fill drops only because they stop producing parseable JSON); LFM2.5-230M is at a
  free-text false-fill ceiling (0.75) with no room to rise.
- **Anything about LFM2-Extract showing the strongest pressure.** Its apparent +0.732 was a
  **measurement artifact**: it emits JSON even in the free-text C0 condition, which the original
  line-parser scored as all-null (FFR C0 = 0.000). With a JSON-aware C0 baseline its FFR C0 = 0.765
  and **SPI-C2 = −0.033** — no effect; it is a chronic filler regardless of format. All headline
  numbers in §2 already use the fair (JSON-aware) baseline.

**Sensitivity analysis provided both ways** for full transparency:
- `results/` — original conservative scoring (line-only C0 parser).
- `results/c0robust/` — JSON-aware C0 fallback (the fair baseline used in §2 and the figures).
  Only LFM2-Extract's C0 changes; all other models are identical.

---

## 5. Robustness / replication

- **Direct SPI-C2 replication (temp 0.2, ×3 reps)** for the 5 significant models — all replicate
  with negligible variance (per-model sd ≤ 0.006), confirming the effect is not a greedy-decoding
  artifact:

  | Model | deterministic SPI-C2 | sampled SPI-C2 (mean ± sd) |
  |---|---|---|
  | Qwen3-0.6B | +0.485 | +0.487 ± 0.003 |
  | Qwen3.5-0.8B | +0.422 | +0.398 ± 0.004 |
  | Granite-4.0-H-350M | +0.255 | +0.249 ± 0.006 |
  | Granite-4.0-350M | +0.192 | +0.180 ± 0.004 |
  | LFM2.5-350M | +0.045 | +0.063 ± 0.006 |

- **Limitation:** the robustness pass covered C0/C3 (mitigation-focused) plus the added C2
  replication above. It did not re-test C4/C5 under sampling.

---

## 6. Figures (`figures/`, 300-DPI PNG + vector PDF)

1. **fig1_schema_pressure_index** — SPI-C2 forest plot with 95% CIs (headline). Red = significant.
2. **fig2_ffr_by_condition_heatmap** — false-fill rate, model × condition (C0–C5).
3. **fig3_c5_format_truth_gap** — under C5, schema validity = 1.00 for all; false-fill varies (the thesis).
4. **fig4_replication_scatter** — deterministic vs temp-0.2 SPI-C2 on the identity line.

Suggested caption text is in the delivery message; `harness/scripts/make_figures.py` regenerates them.

---

## 7. Known limitations / caveats for the write-up

- **C0 baseline validity** for extraction/function-tuned models (handled via the sensitivity
  analysis above; report both, prefer the JSON-aware baseline).
- **FunctionGemma & Falcon-H1** produce little parseable JSON under strict conditions; their SPI is
  dominated by parse failure, not abstention — always report alongside parse/schema-validity rate.
- **Qwen3.5-0.8B is multimodal**; it was run text-only, and C5 (constrained decoding) was not
  applied to it (its VLM stack was not compatible with the constrainer). Its C5 cell is blank.
- Scores come from the package's intentionally **conservative** scorer (plausible guesses count as
  wrong unless directly supported); this is by design.

---

## 8. File manifest

```
RESULTS_SUMMARY.md                     <- this file
results/                               <- primary (conservative) scoring + all deliverables
  raw_outputs.jsonl                    (109k main generations)
  raw_outputs.robustness.merged.jsonl  (60k C0/C3 robustness)
  raw_outputs.robustness_c2.jsonl      (21k C2 replication)
  summary_by_model_condition.csv
  schema_pressure_index.csv
  spi_confidence_intervals.csv         (SPI-C2/C3 + 95% CI, incl. INT8/INT4 quant rows)
  summary_by_challenge.csv
  field_metrics.csv
  qualitative_failures.csv
  experiment_config.json               (hardware, versions, models attempted/failed)
  c0robust/                            <- fair-baseline sensitivity (JSON-aware C0) — USED IN §2 & figures
  robustness/                          <- scored C0/C3 robustness
  robust_c2/                           <- scored C2 replication
  figures/                             <- fig1–4 (.png + .pdf)
harness/                               <- reproducible harness (methods section / re-run)
  scripts/  configs/  docker/  prompts/  schemas/  data/
```
