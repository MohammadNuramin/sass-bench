#!/usr/bin/env python3
"""
SASS-Bench orchestrator.

Runs models x conditions x dataset and writes raw_outputs.jsonl in the schema from
docs/EXPERIMENT_PROTOCOL.md. Resumable, VRAM-safe, records failed models into
experiment_config.json. Does NOT parse/score — that is score_sass_outputs.py's job.

Passes (select with --pass):
  base        C0-C4, deterministic (temp 0). model field = plain id.  -> raw_outputs.jsonl
  quant       C0-C4, INT8 + INT4 for quant_supported models.
              model field = "<id>@int8" / "@int4" so the scorer keeps them separate.
  c5          C5 (= C3 prompt) with schema-constrained decoding (xgrammar/outlines),
              for c5_supported models. model field = plain id, condition "C5".
  robustness  C0 & C3, temp 0.2 / top_p 0.95, 3 reps. model field = "<id>@robust-r{k}"
              so each rep is scored independently (for variance).  -> a separate --out file.

The variant-in-model-field trick lets us reuse score_sass_outputs.py unchanged, since it
groups strictly by (model, condition).
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import platform
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import sass_runner  # noqa: E402

FIELDS = ["target_name", "target_age", "event_name", "city", "event_date"]
COND_KEY = {
    "C0": "C0_free_text",
    "C1": "C1_loose_json",
    "C2": "C2_strict_json",
    "C3": "C3_strict_json_explicit_null_rule",
    "C4": "C4_strict_json_few_shot_null_examples",
    "C5": "C3_strict_json_explicit_null_rule",  # C5 uses the C3 instruction + constrained decoding
}


# --------------------------------------------------------------------------- #
# IO helpers
# --------------------------------------------------------------------------- #
def load_jsonl(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def load_done_keys(out_path):
    """(model, condition, example_id) already written — torn-line tolerant."""
    done = set()
    if os.path.exists(out_path):
        with open(out_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    r = json.loads(line)
                    done.add((r["model"], r["condition"], r["example_id"]))
                except Exception:
                    continue
    return done


def batched(seq, n):
    for i in range(0, len(seq), n):
        yield seq[i : i + n]


def render_prompt(template, ex):
    """Substitute ONLY the two real placeholders. We cannot use str.format() because the
    C4 few-shot template embeds literal JSON braces (e.g. {"target_name":...}) that
    str.format() would misinterpret as replacement fields (KeyError: '"target_name"')."""
    return template.replace("{target_entity}", str(ex["target_entity"])).replace(
        "{context}", str(ex["context"])
    )


def write_json(path, obj):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


# --------------------------------------------------------------------------- #
# experiment_config.json (accumulated across passes)
# --------------------------------------------------------------------------- #
def load_or_init_config(path, args, n_examples):
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            cfg = json.load(f)
    else:
        tmpl = Path("configs/experiment_config_template.json")
        cfg = json.loads(tmpl.read_text(encoding="utf-8")) if tmpl.exists() else {}
    cfg["run_id"] = args.run_id
    cfg["dataset"] = {
        "file": args.data,
        "n_examples": n_examples,
        "seed": 42,
        "generator_script": "scripts/generate_sass_bench.py",
    }
    try:
        import transformers, jsonschema  # noqa

        cfg["software_versions"] = {
            "python": platform.python_version(),
            "transformers": transformers.__version__,
            "torch": torch.__version__,
            "jsonschema": jsonschema.__version__,
            "vllm": None,
            "llama_cpp": None,
        }
    except Exception:
        pass
    gpu = None
    try:
        gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    except Exception:
        pass
    cfg["hardware"] = {
        "gpu": gpu,
        "cpu": platform.processor() or platform.machine(),
        "ram_gb": round(_total_ram_gb(), 1),
        "os": platform.platform(),
    }
    cfg.setdefault("models_attempted", [])
    cfg.setdefault("models_failed", [])
    cfg.setdefault("notes", "")
    return cfg


def _total_ram_gb():
    try:
        import psutil

        return psutil.virtual_memory().total / 1e9
    except Exception:
        try:
            return os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 1e9
        except Exception:
            return 0.0


def record_attempt(cfg, label):
    if label not in cfg["models_attempted"]:
        cfg["models_attempted"].append(label)


def record_failure(cfg, label, stage, err):
    cfg["models_failed"].append({"model": label, "stage": stage, "error": str(err)[:500]})


# --------------------------------------------------------------------------- #
# C5 schema-constrained decoding
# --------------------------------------------------------------------------- #
def _shim_transformers_for_lmfe():
    """lm-format-enforcer's transformers integration imports symbols from old locations
    that moved in transformers 5.x. Re-expose them so its module imports cleanly."""
    import transformers
    import transformers.tokenization_utils as _tu

    # transformers 5.x: PreTrainedTokenizerBase moved to tokenization_utils_base.
    if not hasattr(_tu, "PreTrainedTokenizerBase"):
        _tu.PreTrainedTokenizerBase = transformers.PreTrainedTokenizerBase
    # defensive: LogitsWarper was merged into LogitsProcessor in 5.x.
    if not hasattr(transformers, "LogitsWarper"):
        transformers.LogitsWarper = transformers.LogitsProcessor


def build_c5_processor(lm, schema_dict):
    """Return (extra_generate_kwargs | None, framework_str) for C5 schema-constrained
    decoding. Uses lm-format-enforcer (pure Python: builds a prefix-allowed-tokens
    automaton from the JSON schema). Chosen over xgrammar, whose C++ LogitsProcessor
    hard-segfaults with transformers 5.13 on these models."""
    try:
        _shim_transformers_for_lmfe()
        import lmformatenforcer
        from lmformatenforcer import JsonSchemaParser
        from lmformatenforcer.integrations.transformers import (
            build_transformers_prefix_allowed_tokens_fn,
        )

        parser = JsonSchemaParser(schema_dict)
        prefix_fn = build_transformers_prefix_allowed_tokens_fn(lm.base_tok, parser)
        ver = getattr(lmformatenforcer, "__version__", "?")
        return {"prefix_allowed_tokens_fn": prefix_fn}, f"lm-format-enforcer-{ver}"
    except Exception as e:
        print(f"[c5] lm-format-enforcer unavailable/failed: {e}")
        return None, "none"


# --------------------------------------------------------------------------- #
# Row builder
# --------------------------------------------------------------------------- #
def finalize_row(run_id, model_label, lm, cond, ex, prompt, res, gen_cfg, extra=None):
    row = {
        "run_id": run_id,
        "model": model_label,
        "model_revision": lm.revision,
        "model_parameter_count": sass_runner.human_params(lm.param_count),
        "precision": lm.precision,
        "condition": cond,
        "temperature": gen_cfg.get("temperature", 0),
        "top_p": gen_cfg.get("top_p", 1),
        "max_new_tokens": gen_cfg.get("max_new_tokens", 256),
        "example_id": ex["id"],
        "prompt": prompt,
        "raw_output": res.get("raw_output", ""),
        "parsed_output": None,  # scorer is the single parsing authority
        "parse_success": False,
        "schema_valid": False,
        "latency_ms": res.get("latency_ms"),
        "tokens_in": res.get("tokens_in"),
        "tokens_out": res.get("tokens_out"),
        "used_chat": res.get("used_chat"),
    }
    if lm.param_count_text_only:
        row["model_parameter_count_text_only"] = sass_runner.human_params(lm.param_count_text_only)
    if extra:
        row.update(extra)
    return row


def _cuda_reclaim():
    """Free fragmented VRAM so a post-OOM retry has room. Without this, a single batch
    OOM cascades: the one-by-one retries keep OOMing on the still-fragmented memory."""
    try:
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
    except Exception:
        pass


def safe_generate(lm, prompts, gen_cfg):
    """generate_batch, but on batch failure reclaim VRAM and retry one-by-one so a single
    bad/oom example can't kill the batch."""
    try:
        return sass_runner.generate_batch(lm, prompts, gen_cfg)
    except Exception:
        _cuda_reclaim()  # critical: clear the OOM'd allocation before retrying
        results = []
        for p in prompts:
            try:
                results.append(sass_runner.generate_batch(lm, [p], gen_cfg)[0])
            except Exception as one_err:
                _cuda_reclaim()
                results.append(
                    {
                        "raw_output": "",
                        "tokens_in": None,
                        "tokens_out": None,
                        "latency_ms": None,
                        "used_chat": None,
                        "error": f"{type(one_err).__name__}: {one_err}",
                    }
                )
        return results


# --------------------------------------------------------------------------- #
# Pass planning
# --------------------------------------------------------------------------- #
def plan_variants(manifest, args):
    """Yield (spec, quant, gen_overrides, model_label_suffix, conditions, reps, use_c5)."""
    conds = manifest["conditions"]
    models = manifest["models"]
    if args.models:
        wanted = set(args.models)
        models = [m for m in models if m["model_id"] in wanted or m["short_name"] in wanted]

    if args.pass_ == "base":
        base_conds = args.conditions or conds["base"]
        for spec in models:
            yield dict(spec=spec, quant=None, gen={}, suffix="", conditions=base_conds, reps=1, use_c5=False)

    elif args.pass_ == "quant":
        qcfg = conds["quant"]
        for spec in models:
            if not spec.get("quant_supported"):
                continue
            for variant in qcfg["variants"]:
                yield dict(spec=spec, quant=variant, gen={}, suffix=f"@{variant}",
                           conditions=args.conditions or qcfg["conditions"], reps=1, use_c5=False)

    elif args.pass_ == "c5":
        c5 = conds["c5"]
        for spec in models:
            if not spec.get("c5_supported"):
                continue
            yield dict(spec=spec, quant=None, gen={}, suffix="", conditions=[c5["condition"]],
                       reps=1, use_c5=True)

    elif args.pass_ == "robustness":
        rob = conds["robustness"]
        for spec in models:
            for rep in range(1, rob["repetitions"] + 1):
                yield dict(
                    spec=spec, quant=None,
                    gen={"do_sample": True, "temperature": rob["temperature"],
                         "top_p": rob["top_p"], "seed": 1000 + rep},
                    suffix=f"@robust-r{rep}",
                    conditions=args.conditions or rob["conditions"], reps=1, use_c5=False,
                )
    else:
        raise SystemExit(f"unknown pass {args.pass_}")


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--prompts", default="prompts/prompt_templates.json")
    ap.add_argument("--schema", default="schemas/person_event_v1.schema.json")
    ap.add_argument("--out", required=True)
    ap.add_argument("--config-out", dest="config_out", default="results_placeholder/experiment_config.json")
    ap.add_argument("--run-id", dest="run_id", default="2026-07-03-main")
    ap.add_argument("--pass", dest="pass_", default="base", choices=["base", "quant", "c5", "robustness"])
    ap.add_argument("--conditions", nargs="*", default=None)
    ap.add_argument("--models", nargs="*", default=None)
    ap.add_argument("--max-examples", dest="max_examples", type=int, default=None)
    ap.add_argument("--max-new-tokens", dest="max_new_tokens", type=int, default=256)
    ap.add_argument("--batch-size", dest="batch_size", type=int, default=None,
                    help="override per-model batch_size (e.g. small batch for OOM-prone C5 gap-fill)")
    ap.add_argument("--num-shards", dest="num_shards", type=int, default=1,
                    help="split examples across N machines (strided) for multi-GPU parallelism")
    ap.add_argument("--shard-id", dest="shard_id", type=int, default=0,
                    help="0-indexed shard for this machine; takes examples[shard_id::num_shards]")
    args = ap.parse_args()

    manifest = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    prompts = json.loads(Path(args.prompts).read_text(encoding="utf-8"))
    schema_dict = json.loads(Path(args.schema).read_text(encoding="utf-8"))
    examples = load_jsonl(args.data)
    if args.max_examples is not None:
        examples = examples[: args.max_examples]
    if args.num_shards > 1:
        examples = examples[args.shard_id :: args.num_shards]
        print(f"[shard] shard {args.shard_id}/{args.num_shards}: {len(examples)} examples", flush=True)

    hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    done = load_done_keys(args.out)
    cfg = load_or_init_config(args.config_out, args, len(examples))
    default_bs = manifest["defaults"]["batch_size"]

    out_f = open(args.out, "a", encoding="utf-8", buffering=1)

    print(f"[orchestrate] pass={args.pass_} examples={len(examples)} out={args.out} "
          f"resume_skip={len(done)}", flush=True)

    for plan in plan_variants(manifest, args):
        spec = plan["spec"]
        model_id = spec["model_id"]
        label = model_id + plan["suffix"]
        record_attempt(cfg, label)
        write_json(args.config_out, cfg)

        # Is there any work left for this variant? (skip a full reload if resumed)
        pending_any = any(
            (label, c, ex["id"]) not in done for c in plan["conditions"] for ex in examples
        )
        if not pending_any:
            print(f"[orchestrate] {label}: all done, skipping load", flush=True)
            continue

        # ---- load (with gated fallback) ----
        lm = None
        load_spec = spec
        try:
            lm = sass_runner.load_model(spec, hf_token=hf_token, quant=plan["quant"])
        except Exception as e:
            record_failure(cfg, label, "load", e)
            write_json(args.config_out, cfg)
            fb = spec.get("fallback_model_id")
            if fb and plan["quant"] is None:
                print(f"[orchestrate] {label}: load failed ({e}); trying fallback {fb}", flush=True)
                load_spec = dict(spec, model_id=fb)
                label = fb + plan["suffix"]
                record_attempt(cfg, label)
                try:
                    lm = sass_runner.load_model(load_spec, hf_token=hf_token, quant=plan["quant"])
                except Exception as e2:
                    record_failure(cfg, label, "load-fallback", e2)
                    write_json(args.config_out, cfg)
                    continue
            else:
                continue

        # ---- C5 processor (optional) ----
        gen_base = dict(plan["gen"], max_new_tokens=args.max_new_tokens)
        if plan["use_c5"]:
            c5_extra, fw = build_c5_processor(lm, schema_dict)
            if c5_extra is None:
                record_failure(cfg, label, "c5-constrained", "no schema-constrained decoder available")
                write_json(args.config_out, cfg)
                sass_runner.free_vram(lm)
                lm = None
                continue
            gen_base.update(c5_extra)
            cfg["c5_framework"] = fw
            print(f"[orchestrate] {label}: C5 via {fw}", flush=True)

        bs = args.batch_size or spec.get("batch_size", default_bs)
        n_written = 0
        try:
            for cond in plan["conditions"]:
                template = prompts[COND_KEY[cond]]
                todo = [ex for ex in examples if (label, cond, ex["id"]) not in done]
                gen_cfg = dict(gen_base)
                for chunk in batched(todo, bs):
                    cprompts = [render_prompt(template, ex) for ex in chunk]
                    results = safe_generate(lm, cprompts, gen_cfg)
                    for ex, prompt, res in zip(chunk, cprompts, results):
                        extra = {"error": res["error"]} if res.get("error") else None
                        row = finalize_row(args.run_id, label, lm, cond, ex, prompt, res, gen_cfg, extra)
                        out_f.write(json.dumps(row, ensure_ascii=False) + "\n")
                        done.add((label, cond, ex["id"]))
                        n_written += 1
                    out_f.flush()
                    # Free cached VRAM between batches. Some hybrid models (granite-4.0-h /
                    # Mamba2) otherwise accumulate state across batches until they fill the
                    # 24 GB card and OOM-thrash.
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                print(f"[orchestrate] {label} {cond}: +{len(todo)} (total {n_written})", flush=True)
        except Exception as e:
            record_failure(cfg, label, "generate", e)
        finally:
            sass_runner.free_vram(lm)
            lm = None
            gc.collect()
            write_json(args.config_out, cfg)

    out_f.close()
    write_json(args.config_out, cfg)
    print(f"[orchestrate] pass={args.pass_} complete. config -> {args.config_out}", flush=True)


if __name__ == "__main__":
    main()
