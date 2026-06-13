#!/usr/bin/env bash
# Source this file before configuring/building cuOpt for MetaX C500 via cu-bridge.

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export MACA_PATH="${MACA_PATH:-/opt/maca}"
export CUCC_PATH="${CUCC_PATH:-${MACA_PATH}/tools/cu-bridge}"
export CUCC_TARGETS="${CUCC_TARGETS:-xcore1000}"

# Force MACA to dispatch kernels directly instead of through its deferred/queued
# path. Required for cuOpt on C500: the queued dispatch path causes cross-kernel
# ordering hangs and wrong results (seen in PDLP batch tests run back-to-back).
export MACA_DIRECT_DISPATCH="${MACA_DIRECT_DISPATCH:-1}"

# cu-bridge writes a transient CUDA toolkit shim. Keep it in the workspace so
# builds do not depend on write access to $HOME.
export CUBRIDGE_HOME="${CUBRIDGE_HOME:-${ROOT}/.maca_cu_bridge}"
export WCUDA_HOME="${WCUDA_HOME:-${CUBRIDGE_HOME}}"

if [ -d "${ROOT}/.toolchain/cmake-3.30.8-linux-x86_64/bin" ]; then
  export PATH="${ROOT}/.toolchain/cmake-3.30.8-linux-x86_64/bin:${PATH}"
fi

export PATH="${CUCC_PATH}/tools:${CUCC_PATH}/bin:${MACA_PATH}/bin:${MACA_PATH}/mxgpu_llvm/bin:${PATH}"
export LD_LIBRARY_PATH="${MACA_PATH}/lib:${MACA_PATH}/mxshmem/lib:${MACA_PATH}/ompi/lib:${MACA_PATH}/ucx/lib:/opt/mxdriver/lib:${LD_LIBRARY_PATH:-}"

if [ -d "${ROOT}/.toolchain/boost-install/lib/cmake/Boost-1.84.0" ]; then
  export Boost_DIR="${Boost_DIR:-${ROOT}/.toolchain/boost-install/lib/cmake/Boost-1.84.0}"
fi

if [ -d "${ROOT}/.toolchain/oneapi-tbb-2021.13.0" ]; then
  export TBB_INCLUDE_DIR="${TBB_INCLUDE_DIR:-${ROOT}/.toolchain/oneapi-tbb-2021.13.0/include}"
  export TBB_LIBRARY="${TBB_LIBRARY:-${ROOT}/.toolchain/oneapi-tbb-2021.13.0/lib/intel64/gcc4.8/libtbb.so}"
fi

export BZIP2_ROOT="${BZIP2_ROOT:-/opt/conda}"
export RAPIDS_DATASET_ROOT_DIR="${RAPIDS_DATASET_ROOT_DIR:-${ROOT}/datasets}"

echo "MACA_PATH=${MACA_PATH}"
echo "CUCC_TARGETS=${CUCC_TARGETS}"
echo "CUBRIDGE_HOME=${CUBRIDGE_HOME}"
echo "cmake=$(command -v cmake)"
echo "cmake_maca=$(command -v cmake_maca)"
echo "ninja_maca=$(command -v ninja_maca)"
