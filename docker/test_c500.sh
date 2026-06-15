#!/usr/bin/env bash
#
# Functional tests for the cuopt:c500 image (MetaX C500 / MACA).
# Mirrors docs/dev/build_c500_a100.md sections 3-6. Run INSIDE the container with
# the host home (datasets + mcdf tarball) mounted at /home:
#
#   docker run --name cuopt-c500 -v ~/:/home \
#     --device=/dev/mxcd --device=/dev/dri --gpus all \
#     --cap-add=SYS_ADMIN --cap-add=SYS_PTRACE --security-opt seccomp=unconfined \
#     --shm-size=5g --privileged --group-add video --network=host -it \
#     cuopt:c500 bash /home/cuopt-26.06-maca/docker/test_c500.sh
#
set -uo pipefail
cd /opt/cuopt
source maca_env.sh
export PATH=/opt/cuopt/cpp/build_maca:$PATH            # CLI_TEST spawns cuopt_cli from PATH

DATASETS="${DATASETS:-/home/cuopt-26.06-maca/datasets}"
MCDF_TARBALL="${MCDF_TARBALL:-/home/maca-mcdf-3.7.0.3-linux-x86_64.tar.xz}"
PROXY="${PROXY:-http://127.0.0.1:3578}"
ln -sfn "$DATASETS" /opt/cuopt/datasets
export RAPIDS_DATASET_ROOT_DIR="$DATASETS"

log(){ echo; echo "########## $* ##########"; }
[ -d "$DATASETS" ] || { echo "FATAL: datasets not mounted at $DATASETS"; exit 1; }

############ §3  C++ functional matrix (gtest) ############
B=cpp/build_maca/tests
log "§3 C++ gtest suites (build_maca)"
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

############ §4  Python LP / MILP / QP (numpy-only) ############
log "§4 Python LP/MILP/QP pytest"
PATH=/opt/conda/bin:$PATH PYTHONPATH=python/cuopt RAPIDS_DATASET_ROOT_DIR=$DATASETS \
  /opt/conda/bin/python -m pytest python/cuopt/cuopt/tests/linear_programming -q \
  || echo "pytest LP rc=$?"

############ §5  Python routing + distance (mcdf stack) ############
if [ -f "$MCDF_TARBALL" ]; then
  log "§5 install MetaX mcdf stack"
  PATH=/opt/conda/bin:$PATH bash python/cuopt/setup_maca_rapids.sh "$MCDF_TARBALL" || echo "mcdf setup rc=$?"
  log "§5 Python routing pytest"
  PATH=/opt/conda/bin:$PATH CUDA_PATH=/opt/maca/tools/cu-bridge \
    PYTHONPATH=python/cuopt RAPIDS_DATASET_ROOT_DIR=$DATASETS \
    /opt/conda/bin/python -m pytest python/cuopt/cuopt/tests/routing -q \
    || echo "pytest routing rc=$?"
else
  log "§5 SKIP routing — mcdf tarball not found at $MCDF_TARBALL"
fi

############ §6  REST server smoke ############
log "§6 REST server smoke"
export PATH=/opt/conda/bin:$PATH
export CUDA_PATH=/opt/maca/tools/cu-bridge
export PYTHONPATH=$PWD/python/cuopt:$PWD/python/cuopt_server
python -m cuopt_server.cuopt_service & SRV=$!
for i in $(seq 1 60); do sleep 2
  H=$(curl --noproxy '*' -s http://127.0.0.1:5000/cuopt/health 2>/dev/null) && [ -n "$H" ] && { echo "health: $H"; break; }
done
REQ=$(curl --noproxy '*' -s -X POST http://127.0.0.1:5000/cuopt/request \
  -H "Content-Type: application/json" -d @datasets/cuopt_service_data/cuopt_problem_data.json)
echo "submit -> $REQ"
ID=$(echo "$REQ" | sed -nE 's/.*"reqId":"([^"]+)".*/\1/p')
if [ -n "${ID:-}" ]; then
  for i in $(seq 1 120); do sleep 2     # first solve recompiles MACA kernels (slow)
    curl --noproxy '*' -s -H "Accept: application/msgpack" \
      http://127.0.0.1:5000/cuopt/request/$ID -o /tmp/sol.msgpack 2>/dev/null
    if [ -s /tmp/sol.msgpack ] && ! grep -qi '"reqId"' /tmp/sol.msgpack 2>/dev/null; then
      echo "REST result retrieved ($(wc -c </tmp/sol.msgpack) bytes msgpack)"; break; fi
  done
fi
kill $SRV 2>/dev/null

log "C500 functional tests complete (review output above)"
