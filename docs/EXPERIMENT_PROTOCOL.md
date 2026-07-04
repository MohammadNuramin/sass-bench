# Experiment Protocol

## Goal

Test schema-induced hallucination in newer sub-1B SLMs.

The experiment asks whether a model fills unsupported JSON fields when the correct value should be `null`.

## Benchmark

Use SASS-Bench: Schema-Abstention Stress Benchmark.

Each item contains:

```json
{
  "id": "SASS_000001",
  "domain": "school_event",
  "challenge_type": "MISSING",
  "target_entity": "Lina",
  "context": "...",
  "schema_name": "person_event_v1",
  "gold": {
    "target_name": "Lina",
    "target_age": null,
    "event_name": "robotics workshop",
    "city": "Boston",
    "event_date": "2026-03-14"
  },
  "field_status": {
    "target_name": "SUPPORTED",
    "target_age": "MISSING",
    "event_name": "SUPPORTED",
    "city": "SUPPORTED",
    "event_date": "SUPPORTED"
  },
  "distractors": {}
}
```

## Evidence categories

- SUPPORTED: value is directly stated.
- MISSING: value is absent.
- AMBIGUOUS: value is vague or underspecified.
- CONTRADICTORY: two incompatible values are given.
- DISTRACTOR: a wrong nearby value appears for another entity.

## Model set

Use the primary models in `configs/models_2026_sub1b.json`.

Minimum recommended set:

1. LiquidAI/LFM2.5-230M
2. LiquidAI/LFM2.5-350M
3. Qwen/Qwen3.5-0.8B
4. tiiuae/Falcon-H1-Tiny-R-0.6B
5. ibm-granite/granite-4.0-350m
6. google/functiongemma-270m-it
7. LiquidAI/LFM2-350M-Extract

Add Falcon-H1-Tiny-100M-Instruct if hardware/time allows.

Use only sub-1B models in the main paper. If you include a >1B or exactly 1B model, label it clearly as an external reference and exclude it from primary sub-1B claims.

## Conditions

Run every item under each condition.

### C0 free text

Baseline without JSON schema pressure.

### C1 loose JSON

JSON requested but no explicit strictness.

### C2 strict JSON

Strict JSON, exact keys, no explanation.

### C3 strict JSON + null rule

Strict JSON plus explicit instructions to use `null` when missing, ambiguous, or contradictory.

### C4 strict JSON + few-shot null examples

Strict JSON with demonstrations showing correct null behavior.

### C5 constrained decoding

Same as C3, but enforce JSON schema through a decoding framework. Use only if available.

## Decoding settings

Main run:

```text
temperature = 0
top_p = 1
max_new_tokens = 256
seed = fixed, if supported
```

Secondary robustness run:

```text
temperature = 0.2
top_p = 0.95
repetitions = 3
```

Report the main deterministic run first.

## Required raw output format

Every output must be saved as JSONL. One line per model/condition/example.

```json
{
  "run_id": "2026-07-03-main",
  "model": "LiquidAI/LFM2.5-230M",
  "model_revision": "exact commit hash if possible",
  "model_parameter_count": "0.23B",
  "precision": "BF16/FP16/INT8/INT4/GGUF/etc.",
  "condition": "C3",
  "temperature": 0,
  "top_p": 1,
  "max_new_tokens": 256,
  "example_id": "SASS_000001",
  "prompt": "full prompt string",
  "raw_output": "model output string",
  "parsed_output": null,
  "parse_success": false,
  "schema_valid": false,
  "latency_ms": null,
  "tokens_in": null,
  "tokens_out": null
}
```

The scoring script can parse `raw_output`, but agents should include `parsed_output`, `parse_success`, and `schema_valid` if their runtime already provides them.

## Main metrics

### JSON Parse Rate

Valid JSON outputs divided by total outputs.

### Schema Validity Rate

Outputs with exactly the required keys and type-compatible values.

### Supported Field Accuracy

Accuracy over fields whose gold status is SUPPORTED.

### False Fill Rate

Main metric.

```text
False Fill Rate = false_filled_null_fields / total_gold_null_fields
```

Gold-null fields are MISSING, AMBIGUOUS, and CONTRADICTORY.

### Null Recall

```text
Null Recall = correct_null_outputs / total_gold_null_fields
```

### Abstention Precision

```text
Abstention Precision = correct_null_outputs / total_predicted_null_outputs
```

### Distractor Error Rate

```text
Distractor Error Rate = copied_distractor_fields / total_distractor_fields
```

### Format–Truth Gap

```text
Format–Truth Gap = Schema Validity Rate - Field Faithfulness Rate
```

### Schema Pressure Index

```text
SPI-C2 = False Fill Rate(C2) - False Fill Rate(C0)
SPI-C3 = False Fill Rate(C3) - False Fill Rate(C0)
```

Positive SPI means structured prompting increased unsupported field completion.

## Statistical plan

Use paired bootstrap over examples to estimate confidence intervals for SPI.

Minimum evidence for schema pressure:

```text
SPI-C2 > 0 and 95% bootstrap CI excludes 0.
```

Mitigation evidence:

```text
FFR(C4) < FFR(C3)
```

Also check that supported-field accuracy does not drop by more than 5 percentage points.

## What to return

Return:

```text
summary_by_model_condition.csv
schema_pressure_index.csv
summary_by_challenge.csv
field_metrics.csv
qualitative_failures.csv
experiment_config.json
raw_outputs.jsonl
```
