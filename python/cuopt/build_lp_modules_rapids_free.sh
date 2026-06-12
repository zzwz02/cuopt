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
# Verify:  PYTHONPATH=python/cuopt RAPIDS_DATASET_ROOT_DIR=$PWD/datasets \
#            python -m pytest python/cuopt/cuopt/tests/linear_programming -q
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"   # python/cuopt
ROOT="$(cd "$HERE/../.." && pwd)"                       # repo root
CCCL="${CCCL:-$ROOT/cpp/build/_deps/cccl-src}"
LIBCUOPT_DIR="${LIBCUOPT_DIR:-$ROOT/cpp/build}"
CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"

PYINC="$(python3 -c 'import sysconfig; print(sysconfig.get_path("include"))')"
NPINC="$(python3 -c 'import numpy; print(numpy.get_include())')"
SUF="$(python3 -c 'import sysconfig; print(sysconfig.get_config_var("EXT_SUFFIX"))')"

INCS="-I$PYINC -I$NPINC -I$ROOT/cpp/include -I$ROOT/cpp/vendor/include \
-I$CCCL/libcudacxx/include -I$CCCL/cub -I$CCCL/thrust -I$CUDA_HOME/include"

# C++20 is required: cuopt public headers use std::span.
CXXFLAGS="-std=c++20 -fPIC -O2 -shared -fvisibility=hidden \
-Wno-deprecated-declarations -Wno-unknown-pragmas"
LINK="-L$LIBCUOPT_DIR -lcuopt -L$CUDA_HOME/lib64 -lcudart \
-Wl,-rpath,$LIBCUOPT_DIR -Wl,-rpath,$CUDA_HOME/lib64"

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
  g++ $CXXFLAGS $INCS "$cpp" -o "$HERE/$dir/${base}${SUF}" $LINK
  rm -f "$cpp"
done
echo "ALL LP MODULES BUILT (RAPIDS-free)"
