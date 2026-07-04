#!/usr/bin/env python3
"""
Score SASS-Bench raw outputs.

Input raw_outputs.jsonl: one line per model/condition/example. See docs/EXPERIMENT_PROTOCOL.md.
This script is intentionally conservative: plausible guesses are counted wrong unless supported by gold.
"""
import argparse
import csv
import json
import re
from collections import defaultdict
from pathlib import Path

FIELDS = ["target_name", "target_age", "event_name", "city", "event_date"]
NULL_STRINGS = {
    "", "null", "none", "unknown", "not stated", "not provided",
    "not mentioned", "cannot determine", "can't determine", "n/a", "na",
    "not available", "unspecified", "not specified"
}
GOLD_NULL_STATUSES = {"MISSING", "AMBIGUOUS", "CONTRADICTORY"}

# Set True by --c0-json-fallback. Default False preserves the original conservative scoring.
C0_JSON_FALLBACK = False

def read_jsonl(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows

def normalize_value(value, field):
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)) and field == "target_age":
        try:
            return int(value)
        except Exception:
            return value
    if isinstance(value, str):
        v = value.strip()
        # Strip surrounding quotes sometimes produced in line format.
        if len(v) >= 2 and ((v[0] == v[-1] == '"') or (v[0] == v[-1] == "'")):
            v = v[1:-1].strip()
        low = " ".join(v.lower().split()).rstrip(".")
        if low in NULL_STRINGS:
            return None
        if field == "target_age":
            m = re.fullmatch(r"\d+", low)
            if m:
                return int(low)
            return low
        return low
    return value

def try_parse_json_object(text):
    if text is None:
        return None, False
    if isinstance(text, dict):
        return text, True
    text = str(text).strip()
    # Direct parse.
    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            return obj, True
    except Exception:
        pass

    # Try to extract first JSON object.
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        snippet = text[start:end+1]
        try:
            obj = json.loads(snippet)
            if isinstance(obj, dict):
                return obj, True
        except Exception:
            return None, False
    return None, False

def parse_free_text(text):
    result = {}
    if text is None:
        return result, False
    lines = str(text).splitlines()
    for line in lines:
        if ":" not in line:
            continue
        key, val = line.split(":", 1)
        key = key.strip().lower()
        val = val.strip()
        if key in FIELDS:
            result[key] = val
    return result, all(f in result for f in FIELDS)

def parse_output(row):
    condition = row.get("condition", "")
    raw = row.get("raw_output", "")
    parsed_given = row.get("parsed_output", None)
    if isinstance(parsed_given, dict) and parsed_given:
        return parsed_given, True

    if condition == "C0" or condition.startswith("C0"):
        parsed, ok = parse_free_text(raw)
        # OPT-IN sensitivity fix (default OFF -> original behavior unchanged): extraction-
        # and function-tuned models (e.g. LFM2-Extract) emit JSON even in the free-text C0
        # condition, which the line parser cannot read -> their C0 baseline is scored as
        # all-null and their SPI is spuriously inflated. When enabled, fall back to JSON
        # parsing so C0 gets a fair baseline for those models.
        if (not ok) and C0_JSON_FALLBACK:
            jparsed, jok = try_parse_json_object(raw)
            if jok and jparsed:
                return jparsed, True
        return parsed, ok

    parsed, ok = try_parse_json_object(raw)
    return parsed or {}, ok

def schema_valid(parsed):
    if not isinstance(parsed, dict):
        return False
    if set(parsed.keys()) != set(FIELDS):
        return False
    for f in FIELDS:
        v = parsed.get(f)
        if v is None:
            continue
        if f == "target_age":
            if isinstance(v, int):
                continue
            if isinstance(v, str) and re.fullmatch(r"\d+", v.strip()):
                continue
            # Allow abstention strings as semantic null but not strict schema-valid integer/null.
            return False
        else:
            if not isinstance(v, str):
                return False
    return True

def score_field(gold_value, pred_value, field, field_status, distractor_value=None):
    gold_norm = normalize_value(gold_value, field)
    pred_norm = normalize_value(pred_value, field)

    is_correct = (gold_norm == pred_norm)

    gold_requires_null = field_status in GOLD_NULL_STATUSES
    is_false_fill = bool(gold_requires_null and gold_norm is None and pred_norm is not None)
    is_correct_null = bool(gold_requires_null and gold_norm is None and pred_norm is None)
    predicted_null = pred_norm is None

    is_distractor_error = False
    if distractor_value is not None:
        distractor_norm = normalize_value(distractor_value, field)
        is_distractor_error = bool(pred_norm == distractor_norm and pred_norm != gold_norm)

    return {
        "gold_norm": gold_norm,
        "pred_norm": pred_norm,
        "is_correct": is_correct,
        "is_false_fill": is_false_fill,
        "is_correct_null": is_correct_null,
        "predicted_null": predicted_null,
        "is_distractor_error": is_distractor_error
    }

def pct(num, den):
    return "" if den == 0 else num / den

def write_csv(path, rows, fieldnames):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gold", required=True, help="SASS-Bench JSONL")
    ap.add_argument("--raw", required=True, help="Raw outputs JSONL")
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--c0-json-fallback", dest="c0_json_fallback", action="store_true",
                    help="For C0 free-text: if line parsing fails, fall back to JSON parsing "
                         "(fair baseline for extraction/function-tuned models that emit JSON in C0). "
                         "OFF by default = original conservative scoring.")
    args = ap.parse_args()
    global C0_JSON_FALLBACK
    C0_JSON_FALLBACK = args.c0_json_fallback

    gold_rows = {r["id"]: r for r in read_jsonl(args.gold)}
    raw_rows = read_jsonl(args.raw)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    output_level = []
    field_rows = []

    for row in raw_rows:
        ex_id = row.get("example_id")
        if ex_id not in gold_rows:
            continue
        gold_ex = gold_rows[ex_id]
        parsed, parse_success = parse_output(row)
        sv = schema_valid(parsed) if not row.get("condition", "").startswith("C0") else False

        output_level.append({
            "model": row.get("model"),
            "condition": row.get("condition"),
            "example_id": ex_id,
            "challenge_type": gold_ex.get("challenge_type"),
            "parse_success": parse_success,
            "schema_valid": sv,
        })

        for field in FIELDS:
            pred_value = parsed.get(field) if isinstance(parsed, dict) else None
            gold_value = gold_ex["gold"].get(field)
            field_status = gold_ex["field_status"].get(field)
            distractor_value = gold_ex.get("distractors", {}).get(field)
            s = score_field(gold_value, pred_value, field, field_status, distractor_value)
            field_rows.append({
                "run_id": row.get("run_id"),
                "model": row.get("model"),
                "condition": row.get("condition"),
                "example_id": ex_id,
                "challenge_type": gold_ex.get("challenge_type"),
                "field": field,
                "field_status": field_status,
                "gold_value": json.dumps(gold_value, ensure_ascii=False),
                "predicted_value": json.dumps(pred_value, ensure_ascii=False),
                "gold_norm": json.dumps(s["gold_norm"], ensure_ascii=False),
                "pred_norm": json.dumps(s["pred_norm"], ensure_ascii=False),
                "is_correct": int(s["is_correct"]),
                "is_false_fill": int(s["is_false_fill"]),
                "is_correct_null": int(s["is_correct_null"]),
                "predicted_null": int(s["predicted_null"]),
                "is_distractor_error": int(s["is_distractor_error"]),
                "parse_success": int(parse_success),
                "schema_valid": int(sv)
            })

    write_csv(outdir / "field_metrics.csv", field_rows, [
        "run_id","model","condition","example_id","challenge_type","field","field_status",
        "gold_value","predicted_value","gold_norm","pred_norm","is_correct","is_false_fill",
        "is_correct_null","predicted_null","is_distractor_error","parse_success","schema_valid"
    ])

    # Summaries by model/condition.
    groups = defaultdict(list)
    output_groups = defaultdict(list)
    for r in field_rows:
        groups[(r["model"], r["condition"])].append(r)
    for r in output_level:
        output_groups[(r["model"], r["condition"])].append(r)

    summary = []
    for key, rows in sorted(groups.items()):
        model, condition = key
        outs = output_groups.get(key, [])
        json_den = len(outs)
        parse_rate = pct(sum(1 for o in outs if o["parse_success"]), json_den) if not str(condition).startswith("C0") else ""
        schema_rate = pct(sum(1 for o in outs if o["schema_valid"]), json_den) if not str(condition).startswith("C0") else ""

        supported = [r for r in rows if r["field_status"] == "SUPPORTED"]
        gold_null = [r for r in rows if r["field_status"] in GOLD_NULL_STATUSES]
        predicted_null = [r for r in rows if r["predicted_null"] == 1]
        distractor = [r for r in rows if r["field_status"] == "DISTRACTOR"]

        correct = sum(r["is_correct"] for r in rows)
        faith = pct(correct, len(rows))
        summary.append({
            "model": model,
            "condition": condition,
            "n_outputs": json_den,
            "json_parse_rate": parse_rate,
            "schema_validity_rate": schema_rate,
            "supported_accuracy": pct(sum(r["is_correct"] for r in supported), len(supported)),
            "false_fill_rate": pct(sum(r["is_false_fill"] for r in gold_null), len(gold_null)),
            "null_recall": pct(sum(r["is_correct_null"] for r in gold_null), len(gold_null)),
            "abstention_precision": pct(sum(1 for r in predicted_null if r["field_status"] in GOLD_NULL_STATUSES and r["is_correct_null"] == 1), len(predicted_null)),
            "distractor_error_rate": pct(sum(r["is_distractor_error"] for r in distractor), len(distractor)),
            "field_faithfulness_rate": faith,
            "format_truth_gap": "" if schema_rate == "" or faith == "" else schema_rate - faith
        })

    write_csv(outdir / "summary_by_model_condition.csv", summary, [
        "model","condition","n_outputs","json_parse_rate","schema_validity_rate",
        "supported_accuracy","false_fill_rate","null_recall","abstention_precision",
        "distractor_error_rate","field_faithfulness_rate","format_truth_gap"
    ])

    # Summary by challenge.
    challenge_groups = defaultdict(list)
    for r in field_rows:
        challenge_groups[(r["model"], r["condition"], r["challenge_type"])].append(r)
    challenge_summary = []
    for (model, condition, ch), rows in sorted(challenge_groups.items()):
        gold_null = [r for r in rows if r["field_status"] in GOLD_NULL_STATUSES]
        distractor = [r for r in rows if r["field_status"] == "DISTRACTOR"]
        supported = [r for r in rows if r["field_status"] == "SUPPORTED"]
        challenge_summary.append({
            "model": model,
            "condition": condition,
            "challenge_type": ch,
            "false_fill_rate": pct(sum(r["is_false_fill"] for r in gold_null), len(gold_null)),
            "null_recall": pct(sum(r["is_correct_null"] for r in gold_null), len(gold_null)),
            "distractor_error_rate": pct(sum(r["is_distractor_error"] for r in distractor), len(distractor)),
            "supported_accuracy": pct(sum(r["is_correct"] for r in supported), len(supported)),
            "field_faithfulness_rate": pct(sum(r["is_correct"] for r in rows), len(rows))
        })
    write_csv(outdir / "summary_by_challenge.csv", challenge_summary, [
        "model","condition","challenge_type","false_fill_rate","null_recall",
        "distractor_error_rate","supported_accuracy","field_faithfulness_rate"
    ])

    # Schema pressure index.
    ffr = {(r["model"], r["condition"]): r["false_fill_rate"] for r in summary}
    models = sorted({r["model"] for r in summary})
    spi_rows = []
    for model in models:
        c0 = ffr.get((model, "C0"))
        c2 = ffr.get((model, "C2"))
        c3 = ffr.get((model, "C3"))
        c4 = ffr.get((model, "C4"))
        spi_rows.append({
            "model": model,
            "ffr_c0": c0,
            "ffr_c2": c2,
            "spi_c2": "" if c0 == "" or c2 == "" or c0 is None or c2 is None else c2 - c0,
            "ffr_c3": c3,
            "spi_c3": "" if c0 == "" or c3 == "" or c0 is None or c3 is None else c3 - c0,
            "ffr_c4": c4,
            "few_shot_improvement_c3_minus_c4": "" if c3 == "" or c4 == "" or c3 is None or c4 is None else c3 - c4
        })
    write_csv(outdir / "schema_pressure_index.csv", spi_rows, [
        "model","ffr_c0","ffr_c2","spi_c2","ffr_c3","spi_c3","ffr_c4","few_shot_improvement_c3_minus_c4"
    ])

    # Qualitative failures.
    failures = [r for r in field_rows if r["is_false_fill"] == 1 or r["is_distractor_error"] == 1]
    failures = failures[:200]
    write_csv(outdir / "qualitative_failures.csv", failures, [
        "run_id","model","condition","example_id","challenge_type","field","field_status",
        "gold_value","predicted_value","gold_norm","pred_norm","is_correct","is_false_fill",
        "is_correct_null","predicted_null","is_distractor_error","parse_success","schema_valid"
    ])

    print(f"Wrote results to {outdir}")
    print(f"Scored {len(field_rows)} field rows from {len(output_level)} outputs.")

if __name__ == "__main__":
    main()
