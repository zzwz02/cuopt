#!/usr/bin/env bash
#
# MVP helper: build the LP/MILP Cython extension modules against a RAPIDS-free
# libcuopt.so and place the .so files in the source tree, so the LP/MILP Python
# path can be imported and tested WITHOUT any RAPIDS package (rmm/cudf/pylibraft)
# installed.
#
# This is a stopgap until the python CMake is ported off rapids-cmake /
# rapids-cython (which would build the full wheel, routing included). It builds
# only the 5 modules that `import cuopt.linear_programming` needs.
#
# Prereqs: a built C++ libcuopt.so (cpp/build), CCCL fetched under
# cpp/build/_deps/cccl-src, CUDA toolkit, and `pip install cython`.
#
# Usage:   bash python/cuopt/build_lp_modules_rapids_free.sh
# MACA/C500: source maca_env.sh, put conda python3 on PATH, then
#            CUOPT_RT_MACA=1 LIBCUOPT_DIR=$PWD/cpp/build_maca \
#              bash python/cuopt/build_lp_modules_rapids_free.sh
#            (compiles via cu-bridge pre_make nvcc so cudaMemcpy routes to the
#             MACA runtime; links libruntime_cu/libmcruntime, not NVIDIA cudart)
# Verify:  PYTHONPATH=python/cuopt RAPIDS_DATASET_ROOT_DIR=$PWD/datasets \
#            python -m pytest python/cuopt/cuopt/tests/linear_programming -q
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"   # python/cuopt
ROOT="$(cd "$HERE/../.." && pwd)"                       # repo root
CCCL="${CCCL:-$ROOT/cpp/build/_deps/cccl-src}"
LIBCUOPT_DIR="${LIBCUOPT_DIR:-$ROOT/cpp/build}"
CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"

# MACA/C500 toggle. With CUOPT_RT_MACA=1 the modules are compiled through
# cu-bridge (pre_make nvcc) so the device->host solution copy in the .pyx
# (cudaMemcpy / cudaDeviceSynchronize) is remapped to the MACA runtime at
# compile time, and they are linked against the MACA runtime that owns the
# solver's device memory -- NOT NVIDIA libcudart (which fails on MACA pointers
# and returns garbage/inf solutions). Off by default (NVIDIA build, e.g. A100).
CUOPT_RT_MACA="${CUOPT_RT_MACA:-0}"
MACA_PATH="${MACA_PATH:-/opt/maca}"

PYINC="$(python3 -c 'import sysconfig; print(sysconfig.get_path("include"))')"
NPINC="$(python3 -c 'import numpy; print(numpy.get_include())')"
SUF="$(python3 -c 'import sysconfig; print(sysconfig.get_config_var("EXT_SUFFIX"))')"

INCS="-I$PYINC -I$NPINC -I$ROOT/cpp/include -I$ROOT/cpp/vendor/include \
-I$CCCL/libcudacxx/include -I$CCCL/cub -I$CCCL/thrust -I$CUDA_HOME/include"

if [ "$CUOPT_RT_MACA" = "1" ]; then
  # Use cu-bridge's CCCL (the same MACA CCCL libcuopt was built against) so
  # raft span -> <cuda/std/span> resolves there, and let pre_make nvcc supply
  # cuda_runtime_api.h. Drop the NVIDIA CUDA / cccl-src includes entirely.
  CUBR_INC="$MACA_PATH/tools/cu-bridge/include"
  INCS="-I$PYINC -I$NPINC -I$ROOT/cpp/include -I$ROOT/cpp/vendor/include \
-I$CUBR_INC/cccl -I$CUBR_INC"
  # nvcc front-end: pass host-compiler flags through -Xcompiler.
  CXXFLAGS="-std=c++20 -O2 -shared -Xcompiler -fPIC,-fvisibility=hidden,-Wno-deprecated-declarations,-Wno-unknown-pragmas"
  LINK="-L$LIBCUOPT_DIR -lcuopt -L$MACA_PATH/lib -lruntime_cu -lmcruntime \
-Wl,-rpath,$LIBCUOPT_DIR -Wl,-rpath,$MACA_PATH/lib"
  CXX="/opt/maca/tools/cu-bridge/tools/pre_make nvcc"
else
  # C++20 is required: cuopt public headers use std::span.
  CXXFLAGS="-std=c++20 -fPIC -O2 -shared -fvisibility=hidden \
-Wno-deprecated-declarations -Wno-unknown-pragmas"
  LINK="-L$LIBCUOPT_DIR -lcuopt -L$CUDA_HOME/lib64 -lcudart \
-Wl,-rpath,$LIBCUOPT_DIR -Wl,-rpath,$CUDA_HOME/lib64"
  CXX="g++"
fi

MODS="
cuopt/linear_programming/data_model/data_model_wrapper.pyx
cuopt/linear_programming/solver_settings/solver_settings.pyx
cuopt/linear_programming/io/parser_wrapper.pyx
cuopt/linear_programming/internals/internals.pyx
cuopt/linear_programming/solver/solver_wrapper.pyx
cuopt/routing/vehicle_routing_wrapper.pyx
cuopt/routing/utils_wrapper.pyx
cuopt/distance_engine/waypoint_matrix_wrapper.pyx
"

cd "$HERE"
for pyx in $MODS; do
  base="$(basename "$pyx" .pyx)"; dir="$(dirname "$pyx")"
  cpp="$(mktemp --suffix=.cpp)"
  echo "=== cythonize $pyx ==="
  cython --cplus -3 -I . "$pyx" -o "$cpp"
  echo "=== compile -> $dir/${base}${SUF} ==="
  $CXX $CXXFLAGS $INCS "$cpp" -o "$HERE/$dir/${base}${SUF}" $LINK
  rm -f "$cpp"
done
echo "ALL LP MODULES BUILT (RAPIDS-free)"
