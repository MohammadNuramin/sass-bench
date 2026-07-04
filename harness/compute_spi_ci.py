#!/usr/bin/env python3
"""
Paired-bootstrap 95% confidence intervals for the Schema Pressure Index.

Implements the protocol's statistical plan (docs/EXPERIMENT_PROTOCOL.md):
  SPI-C2 = FFR(C2) - FFR(C0),  SPI-C3 = FFR(C3) - FFR(C0)
  "Minimum evidence for schema pressure: SPI-C2 > 0 and 95% bootstrap CI excludes 0."

Pure post-processing: reads the scorer's field_metrics.csv (never re-parses model output,
so it can never disagree with score_sass_outputs.py). Resamples CLUSTERS (example_id) with
replacement in a paired design — the same resampled examples feed C0/C2/C3 in every replicate,
preserving within-example correlation and the pairing across conditions.

FFR(condition) = sum(is_false_fill over gold-null field rows) / count(gold-null field rows),
gold-null = field_status in {MISSING, AMBIGUOUS, CONTRADICTORY}.

Determinism: numpy Generator is created inside main() from --seed (no import-time randomness,
no datetime), so runs are reproducible and resume-safe.

Variant model labels (e.g. "<id>@int8", "<id>@robust-r1") are treated as independent models,
so this yields CIs for the quantization and robustness variants automatically.
"""
from __future__ import annotations

import argparse
import csv
from collections import defaultdict

import numpy as np

GOLD_NULL = {"MISSING", "AMBIGUOUS", "CONTRADICTORY"}
BASE_COND = "C0"
PRESSURE_CONDS = ["C2", "C3"]


def read_field_metrics(path):
    """Return per (model, condition, example_id): [false_fills, gold_null_count] over gold-null rows,
    plus the set of all example_ids seen per model."""
    agg = defaultdict(lambda: [0, 0])  # (model, cond, ex) -> [false, count]
    model_examples = defaultdict(set)
    with open(path, "r", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            model = r["model"]
            cond = r["condition"]
            ex = r["example_id"]
            model_examples[model].add(ex)
            if r.get("field_status") in GOLD_NULL:
                key = (model, cond, ex)
                agg[key][0] += int(r.get("is_false_fill", 0) or 0)
                agg[key][1] += 1
    return agg, model_examples


def cond_arrays(agg, model, cond, ex_ids):
    """Aligned per-example arrays (false_fills, gold_null_counts) for a condition."""
    f = np.zeros(len(ex_ids), dtype=np.float64)
    d = np.zeros(len(ex_ids), dtype=np.float64)
    for i, ex in enumerate(ex_ids):
        v = agg.get((model, cond, ex))
        if v is not None:
            f[i] = v[0]
            d[i] = v[1]
    return f, d


def ffr(f, d):
    tot = d.sum()
    return (f.sum() / tot) if tot > 0 else float("nan")


def paired_bootstrap(f0, d0, fx, dx, n_boot, rng):
    """Vectorized paired bootstrap of SPI = FFR(x) - FFR(0). Returns array of length n_boot."""
    n = len(f0)
    spis = np.empty(n_boot, dtype=np.float64)
    # Chunk to bound memory of the (chunk, n) index matrix.
    chunk = 2000
    done = 0
    while done < n_boot:
        b = min(chunk, n_boot - done)
        idx = rng.integers(0, n, size=(b, n))
        s_f0 = f0[idx].sum(axis=1)
        s_d0 = d0[idx].sum(axis=1)
        s_fx = fx[idx].sum(axis=1)
        s_dx = dx[idx].sum(axis=1)
        with np.errstate(divide="ignore", invalid="ignore"):
            ffr0 = np.where(s_d0 > 0, s_f0 / s_d0, np.nan)
            ffrx = np.where(s_dx > 0, s_fx / s_dx, np.nan)
        spis[done : done + b] = ffrx - ffr0
        done += b
    return spis


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--field-metrics", dest="field_metrics",
                    default="results_placeholder/field_metrics.csv")
    ap.add_argument("--out", default="results_placeholder/spi_confidence_intervals.csv")
    ap.add_argument("--bootstrap", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=12345)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)  # created here — no import-time randomness
    agg, model_examples = read_field_metrics(args.field_metrics)

    rows = []
    for model in sorted(model_examples):
        ex_ids = sorted(model_examples[model])
        f0, d0 = cond_arrays(agg, model, BASE_COND, ex_ids)
        if d0.sum() == 0:
            continue  # no C0 gold-null data for this model -> can't form SPI
        row = {
            "model": model,
            "n_examples": len(ex_ids),
            "n_gold_null_c0": int(d0.sum()),
            "ffr_c0": ffr(f0, d0),
            "n_bootstrap": args.bootstrap,
            "seed": args.seed,
        }
        for cond in PRESSURE_CONDS:
            fx, dx = cond_arrays(agg, model, cond, ex_ids)
            key = cond.lower()
            if dx.sum() == 0:
                row[f"ffr_{key}"] = ""
                row[f"spi_{key}"] = ""
                row[f"spi_{key}_ci_lo"] = ""
                row[f"spi_{key}_ci_hi"] = ""
                row[f"spi_{key}_significant_positive"] = ""
                continue
            spis = paired_bootstrap(f0, d0, fx, dx, args.bootstrap, rng)
            lo, hi = np.nanpercentile(spis, [2.5, 97.5])
            point = ffr(fx, dx) - ffr(f0, d0)
            row[f"ffr_{key}"] = ffr(fx, dx)
            row[f"spi_{key}"] = point
            row[f"spi_{key}_ci_lo"] = float(lo)
            row[f"spi_{key}_ci_hi"] = float(hi)
            # Protocol decision rule for schema-pressure evidence: SPI>0 and CI excludes 0.
            row[f"spi_{key}_significant_positive"] = bool(lo > 0)
        rows.append(row)

    fieldnames = [
        "model", "n_examples", "n_gold_null_c0", "ffr_c0",
        "ffr_c2", "spi_c2", "spi_c2_ci_lo", "spi_c2_ci_hi", "spi_c2_significant_positive",
        "ffr_c3", "spi_c3", "spi_c3_ci_lo", "spi_c3_ci_hi", "spi_c3_significant_positive",
        "n_bootstrap", "seed",
    ]
    import os
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fieldnames})
    print(f"Wrote {len(rows)} SPI-CI rows to {args.out}")


if __name__ == "__main__":
    main()
