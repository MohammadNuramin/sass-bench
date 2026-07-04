# SASS-Bench execution image.
#
# Base: CUDA 12.4 *devel* (ships nvcc) so the OPTIONAL mamba-ssm / causal-conv1d
# kernels can compile. --gpus all works through Docker Desktop's WSL2 GPU passthrough
# (host driver 591.86 supports cu124). The hybrid Mamba models (falcon_h1,
# granitemoehybrid) run correctly WITHOUT those kernels, just slower — so their
# install is tolerant and never breaks the build.
FROM nvidia/cuda:12.4.1-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    PYTHONUNBUFFERED=1 \
    TOKENIZERS_PARALLELISM=false \
    HF_HOME=/cache \
    HF_HUB_DOWNLOAD_TIMEOUT=60
# NOTE: HF token is deliberately NOT set here — it must never be baked into a layer.
# It is passed only at runtime via -e HF_TOKEN. HF_HUB_ENABLE_HF_TRANSFER is enabled
# in run_all.sh only if hf_transfer imports.

RUN apt-get update && apt-get install -y --no-install-recommends \
      python3.11 python3.11-dev python3.11-venv python3-pip \
      git build-essential ninja-build ca-certificates \
 && ln -sf /usr/bin/python3.11 /usr/bin/python \
 && ln -sf /usr/bin/python3.11 /usr/bin/python3 \
 && rm -rf /var/lib/apt/lists/*

# 1) Torch FIRST, pinned to a cu124 wheel, so the transformers install below does not
#    pull a CPU-only torch as a transitive dependency.
RUN python -m pip install --upgrade pip setuptools wheel \
 && python -m pip install "torch==2.6.*" --index-url https://download.pytorch.org/whl/cu124

# 2) Core deps (transformers >=5.2 is the critical pin). Must succeed.
COPY docker/requirements.lock.txt /tmp/requirements.lock.txt
RUN python -m pip install -r /tmp/requirements.lock.txt

# 3) OPTIONAL add-on deps. Each in its own tolerant layer: a failure here must NOT
#    break the image. The orchestrator records any resulting capability gaps as
#    per-model "failed" entries at run time rather than aborting.
#    - bitsandbytes: INT8/INT4 quantization (RQ6)
#    - lm-format-enforcer: C5 schema-constrained decoding. Pure-Python (prefix-allowed-tokens
#      automaton), chosen over xgrammar whose C++ LogitsProcessor hard-segfaults with
#      transformers 5.13 on these models.
RUN python -m pip install bitsandbytes        || echo "WARN: bitsandbytes skipped (quant add-on unavailable)"
RUN python -m pip install lm-format-enforcer   || echo "WARN: lm-format-enforcer skipped (C5 may be unavailable)"

# 4) OPTIONAL Mamba CUDA kernels — pure speed for the hybrid models (falcon_h1,
#    granitemoehybrid). Compiling from source needs nvcc and 20-40 min; OFF by default
#    because the pure-PyTorch fallback is correct (just ~2x slower for the tiny hybrids).
#    Enable with: docker build --build-arg INSTALL_MAMBA_KERNELS=1
ARG INSTALL_MAMBA_KERNELS=0
ENV MAX_JOBS=8
RUN if [ "$INSTALL_MAMBA_KERNELS" = "1" ]; then \
      (python -m pip install "causal-conv1d>=1.4" || echo "WARN: causal-conv1d skipped") ; \
      (python -m pip install "mamba-ssm>=2.2"     || echo "WARN: mamba-ssm skipped") ; \
    else echo "Mamba kernels off (pure-PyTorch fallback for hybrids)"; fi

# 5) FINAL GUARD: re-pin transformers so no optional dep above could have downgraded it
#    below 5.2 (which would break LFM2.5 "lfm2" / Qwen3.5 "qwen3_5" config loading).
RUN python -m pip install --upgrade "transformers>=5.2" \
 && python -c "import transformers; assert int(transformers.__version__.split('.')[0])>=5, transformers.__version__; print('transformers', transformers.__version__)"

# Non-root. Bind-mounted /cache and /work carry host permissions on Docker Desktop/WSL2.
RUN useradd -m -u 1000 runner && mkdir -p /cache && chown runner:runner /cache
USER runner
WORKDIR /work

CMD ["bash", "scripts/run_all.sh"]
