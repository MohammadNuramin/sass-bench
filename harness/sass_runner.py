#!/usr/bin/env python3
"""
SASS-Bench generalized model runner (library module).

One code path loads all six architecture families present in the primary model list
(lfm2, qwen3_5 [multimodal], qwen3, falcon_h1 [hybrid], granitemoehybrid [hybrid],
gemma3_text) and does batched, deterministic generation on the GPU.

Design contract with score_sass_outputs.py:
  - The runner writes parsed_output=None so the scorer stays the single source of
    parsing/scoring truth (score_sass_outputs.py only trusts parsed_output when it is
    a non-empty dict). We never repair or pre-parse model output before scoring.

Imported by run_experiment.py. Not a CLI on its own.
"""
from __future__ import annotations

import gc
import time
from dataclasses import dataclass, field
from typing import Any

import torch


# --------------------------------------------------------------------------- #
# Loader dispatch
# --------------------------------------------------------------------------- #
KNOWN_VLM_MODEL_TYPES = {"qwen3_5"}


def detect_loader(model_id: str) -> str:
    """Auto-detect which AutoModel class to use, as a fallback when the manifest
    does not pin `loader`. Multimodal / conditional-generation archs -> VLM path."""
    from transformers import AutoConfig

    cfg = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
    archs = list(getattr(cfg, "architectures", None) or [])
    mtype = getattr(cfg, "model_type", "") or ""
    is_vlm = (
        hasattr(cfg, "vision_config")
        or mtype in KNOWN_VLM_MODEL_TYPES
        or any(
            ("ConditionalGeneration" in a) or ("ImageTextToText" in a) or a.endswith("VLM")
            for a in archs
        )
    )
    return "image_text_to_text" if is_vlm else "causal_lm"


def human_params(n: int | None) -> str | None:
    if not n:
        return None
    return f"{n / 1e9:.2f}B"


def _resolve_dtype(dtype: str):
    if dtype in (None, "auto"):
        return "auto"
    return {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}.get(
        dtype, "auto"
    )


def _call_from_pretrained(cls, model_id, hf_token, dtype, quant_config):
    """from_pretrained wrapper that is robust to the transformers 5.x rename of the
    dtype kwarg (torch_dtype -> dtype) and keeps bitsandbytes quant separate."""
    base = dict(trust_remote_code=True, device_map={"": 0}, token=hf_token)
    if quant_config is not None:
        return cls.from_pretrained(model_id, quantization_config=quant_config, **base)
    resolved = _resolve_dtype(dtype)
    try:
        return cls.from_pretrained(model_id, dtype=resolved, **base)
    except TypeError:
        return cls.from_pretrained(model_id, torch_dtype=resolved, **base)


# --------------------------------------------------------------------------- #
# Loaded model handle
# --------------------------------------------------------------------------- #
@dataclass
class LoadedModel:
    model_id: str
    loader: str  # "causal_lm" | "image_text_to_text"
    model: Any
    tok: Any  # tokenizer OR processor
    is_processor: bool
    base_tok: Any  # the underlying tokenizer (processor.tokenizer or tok)
    revision: str | None
    param_count: int
    param_count_text_only: int | None
    precision: str
    chat_kwargs: dict = field(default_factory=dict)
    force_no_cache: bool = False  # set True after a hybrid-cache generate error (e.g. granitemoehybrid)


def _language_model_param_count(model) -> int | None:
    """Best-effort text-tower param count for VLMs, so the sub-1B claim stays honest."""
    for attr_path in ("language_model", "model.language_model", "model.text_model", "text_model"):
        obj = model
        try:
            for part in attr_path.split("."):
                obj = getattr(obj, part)
            if obj is not None and hasattr(obj, "parameters"):
                return sum(p.numel() for p in obj.parameters())
        except AttributeError:
            continue
    return None


def load_model(spec: dict, hf_token: str | None = None, quant: str | None = None) -> LoadedModel:
    """Load one model. `quant` in {None, "int8", "int4"} enables bitsandbytes.

    Raises on failure — the orchestrator catches and records the failure.
    """
    from transformers import (
        AutoModelForCausalLM,
        AutoModelForImageTextToText,
        AutoProcessor,
        AutoTokenizer,
    )

    model_id = spec["model_id"]
    loader = spec.get("loader") or detect_loader(model_id)
    dtype = spec.get("dtype", "auto")

    quant_config = None
    if quant in ("int8", "int4"):
        from transformers import BitsAndBytesConfig

        if quant == "int8":
            quant_config = BitsAndBytesConfig(load_in_8bit=True)
        else:
            quant_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
            )

    if loader == "image_text_to_text":
        # Text-only VLM usage: prefer AutoProcessor, but fall back to AutoTokenizer when the
        # processor needs an image stack (e.g. Qwen2VLImageProcessor -> torchvision) that we
        # don't need for text extraction. The Qwen chat template lives in tokenizer_config too.
        try:
            tok = AutoProcessor.from_pretrained(model_id, trust_remote_code=True, token=hf_token)
            is_proc = True
            base_tok = getattr(tok, "tokenizer", tok)
        except Exception as proc_err:
            print(f"[sass_runner] AutoProcessor failed for {model_id} ({proc_err}); "
                  f"falling back to AutoTokenizer (text-only)")
            tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True, token=hf_token)
            is_proc = False
            base_tok = tok
        model = _call_from_pretrained(AutoModelForImageTextToText, model_id, hf_token, dtype, quant_config)
    else:
        tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True, token=hf_token)
        model = _call_from_pretrained(AutoModelForCausalLM, model_id, hf_token, dtype, quant_config)
        is_proc = False
        base_tok = tok

    # Left-padding is mandatory for correct batched decoder-only generation.
    base_tok.padding_side = "left"
    if base_tok.pad_token_id is None:
        base_tok.pad_token = base_tok.eos_token

    model.eval()

    revision = None
    try:
        import huggingface_hub

        revision = huggingface_hub.model_info(model_id, token=hf_token).sha
    except Exception:
        revision = None

    param_count = sum(p.numel() for p in model.parameters())
    param_text_only = _language_model_param_count(model) if is_proc else None

    if quant:
        precision = quant
    else:
        precision = str(getattr(model, "dtype", dtype)).replace("torch.", "")

    return LoadedModel(
        model_id=model_id,
        loader=loader,
        model=model,
        tok=tok,
        is_processor=is_proc,
        base_tok=base_tok,
        revision=revision,
        param_count=param_count,
        param_count_text_only=param_text_only,
        precision=precision,
        chat_kwargs=dict(spec.get("chat_kwargs", {})),
    )


# --------------------------------------------------------------------------- #
# Prompt building + generation
# --------------------------------------------------------------------------- #
def build_inputs(lm: LoadedModel, prompts: list[str]):
    """Apply the chat template (text-only) with graceful fallbacks. Returns
    (encoded_dict_on_device, used_chat_bool)."""
    device = lm.model.device

    def _to_device(enc):
        return {k: v.to(device) for k, v in enc.items() if hasattr(v, "to")}

    # Tier 1: chat template WITH per-model chat_kwargs (e.g. enable_thinking=False).
    for attempt_kwargs in (lm.chat_kwargs, {}):
        try:
            if lm.is_processor:
                convos = [[{"role": "user", "content": [{"type": "text", "text": p}]}] for p in prompts]
            else:
                convos = [[{"role": "user", "content": p}] for p in prompts]
            enc = lm.tok.apply_chat_template(
                convos,
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
                padding=True,
                **attempt_kwargs,
            )
            # Keep only what decoder-only generation needs; drop token_type_ids and any
            # image/modality tensors (some model forwards reject unexpected kwargs).
            enc = {k: v for k, v in enc.items() if k in ("input_ids", "attention_mask")}
            return _to_device(enc), True
        except Exception:
            continue

    # Tier 3: raw tokenization fallback.
    enc = lm.base_tok(prompts, return_tensors="pt", padding=True)
    return _to_device(enc), False


def _count_to_eos(row: torch.Tensor, eos_id: int | None, pad_id: int | None) -> int:
    """Count generated tokens up to (and including) the first EOS; ignore trailing pad."""
    n = 0
    for tok_id in row.tolist():
        if pad_id is not None and tok_id == pad_id and eos_id != pad_id:
            break
        n += 1
        if eos_id is not None and tok_id == eos_id:
            break
    return n


def generate_batch(lm: LoadedModel, prompts: list[str], gen_cfg: dict) -> list[dict]:
    """Deterministic (or sampled) batched generation.

    gen_cfg keys: max_new_tokens, do_sample, temperature, top_p, seed.
    Returns one dict per prompt: raw_output, tokens_in, tokens_out, latency_ms, used_chat.
    """
    enc, used_chat = build_inputs(lm, prompts)
    in_len = enc["input_ids"].shape[1]
    base_tok = lm.base_tok
    eos_id = base_tok.eos_token_id
    pad_id = base_tok.pad_token_id

    do_sample = bool(gen_cfg.get("do_sample", False))
    seed = gen_cfg.get("seed", 0)
    if seed is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    gk: dict[str, Any] = dict(
        max_new_tokens=gen_cfg.get("max_new_tokens", 256),
        do_sample=do_sample,
        num_beams=1,
        pad_token_id=pad_id,
    )
    if do_sample:
        gk["temperature"] = gen_cfg.get("temperature", 0.2)
        gk["top_p"] = gen_cfg.get("top_p", 0.95)
    # else: greedy == temperature 0 / top_p 1; HF ignores temp/top_p when not sampling.
    if gen_cfg.get("logits_processor") is not None:
        gk["logits_processor"] = gen_cfg["logits_processor"]
    if gen_cfg.get("prefix_allowed_tokens_fn") is not None:
        # C5 schema-constrained decoding via lm-format-enforcer (pure-Python automaton).
        gk["prefix_allowed_tokens_fn"] = gen_cfg["prefix_allowed_tokens_fn"]
    if lm.force_no_cache:
        gk["use_cache"] = False  # sticky after a prior hybrid-cache failure for this model

    t0 = time.perf_counter()
    with torch.inference_mode():
        try:
            out = lm.model.generate(**enc, **gk)
        except Exception as e1:
            # Fix 1: hybrid-cache errors (e.g. granitemoehybrid "has_previous_state can only
            # be called on LinearAttention layers") — retry once with cache disabled, then
            # make it sticky for this model so later batches skip the failed attempt.
            try:
                out = lm.model.generate(**{**enc}, **{**gk, "use_cache": False})
                lm.force_no_cache = True
            except Exception:
                # Fix 2: VLM fallback — some conditional-generation models route text
                # through a language_model submodule.
                lm_sub = getattr(lm.model, "language_model", None)
                if lm_sub is not None and hasattr(lm_sub, "generate"):
                    out = lm_sub.generate(**enc, **gk)
                else:
                    raise e1
    dt_ms = (time.perf_counter() - t0) * 1000.0

    gen = out[:, in_len:]  # uniform slice thanks to left-padding
    texts = base_tok.batch_decode(gen, skip_special_tokens=True)
    tokens_in = enc["attention_mask"].sum(dim=1).tolist()
    tokens_out = [_count_to_eos(gen[i], eos_id, pad_id) for i in range(gen.shape[0])]
    per_ms = dt_ms / max(len(prompts), 1)

    return [
        {
            "raw_output": t.strip(),
            "tokens_in": int(ti),
            "tokens_out": int(to),
            "latency_ms": round(per_ms, 2),
            "used_chat": used_chat,
        }
        for t, ti, to in zip(texts, tokens_in, tokens_out)
    ]


def free_vram(lm: LoadedModel | None) -> None:
    """Release a model's GPU memory before loading the next one."""
    if lm is not None:
        try:
            del lm.model
            del lm.tok
            del lm.base_tok
        except Exception:
            pass
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
