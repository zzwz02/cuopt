#!/usr/bin/env bash
#
# Functional tests for the cuopt:a100 image (NVIDIA A100 / CUDA 12.9).
# Mirrors docs/dev/build_c500_a100.md sections 3-6. Run INSIDE the container with
# the host home (datasets) mounted at /home and the NVIDIA GPU exposed:
#
#   docker run --name cuopt-a100 -v ~/:/home --gpus all \
#     --cap-add=SYS_ADMIN --cap-add=SYS_PTRACE --security-opt seccomp=unconfined \
#     --shm-size=5g --network=host -it \
#     cuopt:a100 bash /home/cuopt-26.06-maca/docker/test_a100.sh
#
# Native NVIDIA RAPIDS (cudf 24.12) is baked into /opt/cuopt/.a100-venv at image build
# time, and the cuOpt Cython modules are already compiled against that venv's numpy, so
# the whole matrix (LP/MILP/QP, routing/distance, REST) runs straight from the venv —
# no install or rebuild at test time.
#
set -uo pipefail
cd /opt/cuopt
VENV=/opt/cuopt/.a100-venv
PY="$VENV/bin/python"
export PATH="$VENV/bin:/opt/cuopt/cpp/build_cuda:$PATH"
export LD_LIBRARY_PATH=/opt/cuopt/cpp/build_cuda:/usr/lib/x86_64-linux-gnu:/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}
export CUDA_HOME=/usr/local/cuda
# numba grabs the CUDA toolkit stub libcuda otherwise (CUDA_ERROR_STUB_LIBRARY) — point
# it at the real driver injected by --gpus all.
export NUMBA_CUDA_DRIVER=/usr/lib/x86_64-linux-gnu/libcuda.so.1

DATASETS="${DATASETS:-/home/cuopt-26.06-maca/datasets}"
ln -sfn "$DATASETS" /opt/cuopt/datasets
export RAPIDS_DATASET_ROOT_DIR="$DATASETS"

log(){ echo; echo "########## $* ##########"; }
[ -d "$DATASETS" ] || { echo "FATAL: datasets not mounted at $DATASETS"; exit 1; }

############ §3  C++ functional matrix (gtest) ############
B=cpp/build_cuda/tests
log "§3 C++ gtest suites (build_cuda)"
for t in \
  linear_programming/LP_UNIT_TEST linear_programming/PDLP_TEST \
  linear_programming/MPS_PARSER_TEST linear_programming/C_API_TEST \
  dual_simplex/DUAL_SIMPLEX_TEST qp/QP_UNIT_TEST socp/SOCP_TEST \
  mip/MIP_TEST mip/MIP_TERMINATION_STATUS_TEST mip/PRESOLVE_TEST mip/INCUMBENT_CALLBACK_TEST \
  routing/ROUTING_UNIT_TEST distance_engine/WAYPOINT_MATRIXTEST utilities/CLI_TEST ; do
  echo "== $t =="; "$B/$t" --gtest_brief=1 || echo "RC=$? ($t)"
done

log "§3 examples"
for e in cvrp_daily_deliveries pdptw_mixed_fleet service_team_routing ; do
  echo "== example $e =="; "$B/examples/routing/$e" || echo "RC=$? ($e)"
done

############ §4  Python LP / MILP / QP ############
log "§4 Python LP/MILP/QP pytest"
PYTHONPATH=python/cuopt RAPIDS_DATASET_ROOT_DIR=$DATASETS \
  "$PY" -m pytest python/cuopt/cuopt/tests/linear_programming -q || echo "pytest LP rc=$?"

############ §5  Python routing + distance (native RAPIDS) ############
log "§5 Python routing pytest (native NVIDIA RAPIDS)"
PYTHONPATH=python/cuopt RAPIDS_DATASET_ROOT_DIR=$DATASETS \
  "$PY" -m pytest python/cuopt/cuopt/tests/routing -q || echo "pytest routing rc=$?"

############ §6  REST server smoke ############
log "§6 REST server smoke"
export PYTHONPATH=$PWD/python/cuopt:$PWD/python/cuopt_server
"$PY" -m cuopt_server.cuopt_service & SRV=$!
for i in $(seq 1 45); do sleep 2
  H=$(curl --noproxy '*' -s http://127.0.0.1:5000/cuopt/health 2>/dev/null) && [ -n "$H" ] && { echo "health: $H"; break; }
done
REQ=$(curl --noproxy '*' -s -X POST http://127.0.0.1:5000/cuopt/request \
  -H "Content-Type: application/json" -d @datasets/cuopt_service_data/cuopt_problem_data.json)
echo "submit -> $REQ"
ID=$(echo "$REQ" | sed -nE 's/.*"reqId":"([^"]+)".*/\1/p')
if [ -n "${ID:-}" ]; then
  for i in $(seq 1 60); do sleep 2
    curl --noproxy '*' -s -H "Accept: application/msgpack" \
      http://127.0.0.1:5000/cuopt/request/$ID -o /tmp/sol.msgpack 2>/dev/null
    if [ -s /tmp/sol.msgpack ] && ! grep -qi '"reqId"' /tmp/sol.msgpack 2>/dev/null; then
      echo "REST result retrieved ($(wc -c </tmp/sol.msgpack) bytes msgpack)"; break; fi
  done
fi
kill $SRV 2>/dev/null

log "A100 functional tests complete (review output above)"
