#!/usr/bin/env bash
# Capture Nsight Systems profiles of the cuOpt LP path on the A100 (build_cuda),
# NVTX-enabled (-DCUOPT_ENABLE_NVTX). One .nsys-rep per (instance, method).
#   method 1 = PDLP (first-order)      method 3 = barrier (interior point)
#
# Phase NVTX ranges now laid out back-to-back:
#   H2D: upload problem to device  (cuopt_cli, before the solve)
#   LP phase: problem check | problem build | presolve | PDLP solve | postsolve
#   D2H: solution to host + write .sol  (only with --solution-file)
# --eager-module-loading forces CUDA_MODULE_LOADING=EAGER for method 1/3 too, so
# "Runtime Triggered Module Loading" is front-loaded out of the solve timeline.
set -uo pipefail

CLI=/home/cuopt-26.06-maca/cpp/build_cuda/cuopt_cli
OUT=/home/cuopt-26.06-maca/nsys_profiles
DATA=/home/cuopt-26.06-maca/datasets/linear_programming
export CUDA_VISIBLE_DEVICES=0          # A100-PCIE-40GB (only NVIDIA device)
TRACE="cuda,nvtx,cublas,cusparse,osrt"
TL=${TIME_LIMIT:-600}                  # safety cap (s); override: TIME_LIMIT=60 ./run_profiles.sh
mkdir -p "$OUT"

run() {  # <instance> <method> <tag>
  local inst=$1 method=$2 tag=$3
  local mps="$DATA/$inst/$inst.mps"
  echo "=== nsys: $inst  method=$method ($tag) ==="
  nsys profile \
    --output="$OUT/${inst}_${tag}_m${method}" \
    --force-overwrite=true \
    --trace="$TRACE" \
    --sample=none \
    "$CLI" "$mps" --method "$method" --time-limit "$TL" \
      --eager-module-loading
  echo
}

run scpm1       1 pdlp
run scpm1       3 barrier
run woodlands09 1 pdlp
run woodlands09 3 barrier

echo "=== artifacts ==="
ls -la "$OUT"/*.nsys-rep
