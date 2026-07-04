#!/usr/bin/env bash
# SASS-Bench end-to-end entrypoint (runs inside the Docker container).
#
# Usage:  bash scripts/run_all.sh [STAGE]
#   STAGE = preflight | smoke | base | quant | c5 | robustness | score | full | all
#   default: smoke
#
# Env overrides:
#   RUN_ID   (default 2026-07-03-main)     N (dataset size, default 1000)
#   SEED     (default 42)                  SMOKE_N (default 20)
#   BOOTSTRAP (default 10000)
#
# The orchestrator handles per-model failures internally (records them in
# experiment_config.json and continues), so a single model dying never aborts the run.
set -uo pipefail

STAGE="${1:-smoke}"
RUN_ID="${RUN_ID:-2026-07-03-main}"
N="${N:-1000}"
SEED="${SEED:-42}"
SMOKE_N="${SMOKE_N:-20}"
BOOTSTRAP="${BOOTSTRAP:-10000}"

MANIFEST="configs/run_manifest.json"
DATA="data/sass_bench.jsonl"
RES="results_placeholder"
MAIN_RAW="$RES/raw_outputs.jsonl"
ROBUST_RAW="$RES/raw_outputs.robustness.jsonl"
SMOKE_RAW="$RES/raw_outputs.smoke.jsonl"

log(){ echo "[run_all] $*"; }

# --- enable fast Hub downloads only if hf_transfer is importable ---
if python -c "import hf_transfer" 2>/dev/null; then
  export HF_HUB_ENABLE_HF_TRANSFER=1
  log "hf_transfer enabled"
fi

preflight(){
  log "preflight: python=$(python -V 2>&1)"
  python - <<'PY'
import sys, torch, transformers
maj = int(transformers.__version__.split('.')[0])
print(f"[preflight] torch={torch.__version__} cuda_available={torch.cuda.is_available()}")
print(f"[preflight] transformers={transformers.__version__}")
if torch.cuda.is_available():
    print(f"[preflight] gpu={torch.cuda.get_device_name(0)}")
assert torch.cuda.is_available(), "CUDA not visible — run with --gpus all"
assert maj >= 5, f"transformers {transformers.__version__} < 5.2 (LFM2.5/Qwen3.5 need >=5.2)"
print("[preflight] OK")
PY
}

gen_data(){
  if [ ! -f "$DATA" ]; then
    log "generating dataset n=$N seed=$SEED -> $DATA"
    python scripts/generate_sass_bench.py --n "$N" --seed "$SEED" --out "$DATA"
  else
    log "dataset exists: $DATA ($(wc -l < "$DATA") rows)"
  fi
}

score_and_ci(){
  local raw="$1"; local outdir="$2"
  log "scoring $raw -> $outdir"
  python scripts/score_sass_outputs.py --gold "$DATA" --raw "$raw" --outdir "$outdir"
  log "bootstrap CIs (B=$BOOTSTRAP)"
  python scripts/compute_spi_ci.py --field-metrics "$outdir/field_metrics.csv" \
    --out "$outdir/spi_confidence_intervals.csv" --bootstrap "$BOOTSTRAP" --seed 12345
}

run_pass(){ # pass out [extra args...]
  local p="$1"; local out="$2"; shift 2
  log "pass=$p -> $out"
  python scripts/run_experiment.py --manifest "$MANIFEST" --data "$DATA" \
    --out "$out" --config-out "$RES/experiment_config.json" --run-id "$RUN_ID" \
    --pass "$p" "$@"
}

case "$STAGE" in
  preflight) preflight ;;

  smoke)
    preflight; gen_data
    run_pass base "$SMOKE_RAW" --run-id "${RUN_ID}-smoke" --max-examples "$SMOKE_N"
    run_pass quant "$SMOKE_RAW" --run-id "${RUN_ID}-smoke" --max-examples "$SMOKE_N" --models "LiquidAI/LFM2.5-230M"
    run_pass c5 "$SMOKE_RAW" --run-id "${RUN_ID}-smoke" --max-examples "$SMOKE_N" --models "LiquidAI/LFM2.5-230M"
    score_and_ci "$SMOKE_RAW" "$RES/smoke"
    log "SMOKE DONE — inspect $RES/smoke/summary_by_model_condition.csv and experiment_config.json"
    ;;

  base)       preflight; gen_data; run_pass base "$MAIN_RAW" ;;
  quant)      preflight; gen_data; run_pass quant "$MAIN_RAW" ;;
  c5)         preflight; gen_data; run_pass c5 "$MAIN_RAW" ;;
  robustness) preflight; gen_data; run_pass robustness "$ROBUST_RAW" ;;

  score)
    score_and_ci "$MAIN_RAW" "$RES"
    [ -f "$ROBUST_RAW" ] && score_and_ci "$ROBUST_RAW" "$RES/robustness"
    ;;

  full|all)
    preflight; gen_data
    run_pass base  "$MAIN_RAW"
    run_pass quant "$MAIN_RAW"
    run_pass c5    "$MAIN_RAW"
    run_pass robustness "$ROBUST_RAW"
    score_and_ci "$MAIN_RAW" "$RES"
    score_and_ci "$ROBUST_RAW" "$RES/robustness"
    log "FULL RUN DONE — bundle in $RES/"
    ;;

  *) echo "unknown stage: $STAGE"; exit 2 ;;
esac
