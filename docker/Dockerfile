# syntax=docker/dockerfile:1
#
# cuOpt — from-source build for MetaX C500 (MACA/cu-bridge) and NVIDIA A100 (CUDA 12.9).
# One Dockerfile, two targets -> two images:
#     docker build --target c500 -t cuopt:c500 ...
#     docker build --target a100 -t cuopt:a100 ...
#
# Constraints honored:
#   * NOTHING is COPYed from the build context. Source (cuOpt fork), CCCL, and CMake
#     are all cloned/downloaded during the build.
#   * Toolchain via apt where possible (ninja/boost/tbb/bz2); CMake>=3.30 and CCCL are
#     downloaded (binary / git) since apt/jammy can't provide them.
#   * Datasets are NOT baked in — mount them at run time (see docker/test_*.sh).
#
# Base image already provides: CUDA 12.9 (/usr/local/cuda), MACA + cu-bridge (/opt/maca,
# cmake_maca/ninja_maca/mxcc), conda python3.10 (/opt/conda), ninja, git, g++.
#
# Networking: apt uses the in-image MetaX mirror (direct), pip uses the Aliyun mirror
# (direct). Only github access (clones + CMake FetchContent of papilo/pslp/gtest) needs
# the proxy, passed via --build-arg PROXY and applied inline per-RUN. Build with
# `--network=host` so the proxy at 127.0.0.1 is reachable.

ARG BASE_IMAGE=sw-dockhub-zj.mxcr.io/xiangrong/ai-agent:20260506-690-torch2.4-py310-ubuntu22.04-amd64

############################################################
# Stage: common — toolchain + a fresh cuOpt source clone
############################################################
FROM ${BASE_IMAGE} AS common

ARG PROXY=
ARG CUOPT_REPO=https://github.com/zzwz02/cuopt.git
ARG CUOPT_REF=feature/maca-port
ARG PIP_INDEX=http://mirrors.aliyun.com/pypi/simple/
ARG PIP_HOST=mirrors.aliyun.com

SHELL ["/bin/bash", "-c"]
ENV DEBIAN_FRONTEND=noninteractive

# --- build tools via apt (MetaX mirror already configured in the base; no proxy) ---
RUN apt-get update && apt-get install -y --no-install-recommends \
        ninja-build \
        libboost-all-dev \
        libtbb-dev \
        libbz2-dev \
        zlib1g-dev \
        libsuitesparse-dev \
        pkg-config \
        patchelf \
        ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# --- CMake 3.30.8 (system cmake is 3.22; cuOpt requires >=3.30). Official binary. ---
RUN http_proxy=${PROXY} https_proxy=${PROXY} \
      curl -fsSL -o /tmp/cmake.tgz \
      https://github.com/Kitware/CMake/releases/download/v3.30.8/cmake-3.30.8-linux-x86_64.tar.gz \
    && tar -C /opt -xf /tmp/cmake.tgz \
    && rm /tmp/cmake.tgz
ENV PATH=/opt/cmake-3.30.8-linux-x86_64/bin:${PATH}

# --- Python build deps into conda python3.10 (Aliyun mirror; no proxy) ---
RUN /opt/conda/bin/python -m pip install --no-cache-dir \
        -i ${PIP_INDEX} --trusted-host ${PIP_HOST} \
        cython numpy

# --- CCCL 3.4.x source for the A100 build. cuOpt's get_cccl.cmake pins GIT_TAG v3.4.0,
#     which has no public tag; the 3.4.x release branch provides matching headers and is
#     injected via -DFETCHCONTENT_SOURCE_DIR_CCCL. (C500 uses MACA's own CCCL, no fetch.) ---
RUN http_proxy=${PROXY} https_proxy=${PROXY} \
      git clone --depth 1 --branch branch/3.4.x \
      https://github.com/NVIDIA/cccl.git /opt/cccl-src

# --- fresh cuOpt source from the fork remote ---
RUN http_proxy=${PROXY} https_proxy=${PROXY} \
      git clone --depth 1 --branch ${CUOPT_REF} ${CUOPT_REPO} /opt/cuopt

WORKDIR /opt/cuopt

############################################################
# Stage: c500 — MACA / cu-bridge build (cpp/build_maca)
############################################################
FROM common AS c500
ARG PROXY=

# C++ core + CLI + all gtest binaries, built through cu-bridge.
# maca_env.sh exports MACA_PATH / cu-bridge PATH / MACA_DIRECT_DISPATCH / CUBRIDGE_HOME.
# TBB/Boost come from apt: pass TBB paths explicitly (custom FindTBB reads cache vars);
# Boost 1.74 is auto-found by papilo's find_package(Boost) under /usr.
RUN source maca_env.sh \
    && http_proxy=${PROXY} https_proxy=${PROXY} \
       cmake_maca -S cpp -B cpp/build_maca -G Ninja \
         -DCUOPT_MACA=ON \
         -DCMAKE_BUILD_TYPE=Release \
         -DCMAKE_CUDA_ARCHITECTURES=80 \
         -DBUILD_TESTS=ON \
         -DSKIP_GRPC_BUILD=ON \
         -DTBB_INCLUDE_DIR=/usr/include \
         -DTBB_LIBRARY=/usr/lib/x86_64-linux-gnu/libtbb.so \
    && ninja_maca -C cpp/build_maca -j "$(nproc)"

# Python LP/MILP/QP + routing/distance Cython modules, compiled via cu-bridge so the
# .pyx device->host copy routes to the MACA runtime (CUOPT_RT_MACA=1).
RUN source maca_env.sh \
    && export PATH=/opt/conda/bin:${PATH} \
    && CUOPT_RT_MACA=1 LIBCUOPT_DIR=/opt/cuopt/cpp/build_maca \
       bash python/cuopt/build_lp_modules_rapids_free.sh

# Python runtime deps for LP/MILP/QP tests (cuopt) + REST server (cuopt_server), conda env.
# numpy pinned to the build-time version so the compiled .so keep ABI compatibility.
# cupy is intentionally OMITTED on C500 — MetaX mcpy provides it (installed by
# setup_maca_rapids.sh at test time); real cupy-cuda12x would break on MACA.
ARG PIP_INDEX=http://mirrors.aliyun.com/pypi/simple/
ARG PIP_HOST=mirrors.aliyun.com
RUN /opt/conda/bin/python -m pip install --no-cache-dir \
        -i ${PIP_INDEX} --trusted-host ${PIP_HOST} \
        'numpy==1.26.4' 'pandas>=2.0' 'pyyaml>=6.0.0' 'scipy>=1.14.1' \
        'fastapi' 'jsonref==1.1.0' 'msgpack-numpy==0.4.8' 'msgpack==1.1.2' \
        'psutil>=6.0.0' 'uvicorn==0.34.*'

############################################################
# Stage: a100 — NVIDIA CUDA 12.9 build (cpp/build_cuda)
############################################################
FROM common AS a100
ARG PROXY=

# C++ core + CLI + all gtest binaries with the standard NVIDIA toolchain.
# CCCL injected from the local 3.4.x clone (no public v3.4.0 tag for FetchContent).
RUN http_proxy=${PROXY} https_proxy=${PROXY} \
      cmake -S cpp -B cpp/build_cuda -G Ninja \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_CUDA_ARCHITECTURES=80 \
        -DBUILD_TESTS=ON \
        -DSKIP_GRPC_BUILD=ON \
        -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc \
        -DTBB_INCLUDE_DIR=/usr/include \
        -DTBB_LIBRARY=/usr/lib/x86_64-linux-gnu/libtbb.so \
        -DRAPIDS_DATASET_ROOT_DIR=/opt/cuopt/datasets \
        -DFETCHCONTENT_SOURCE_DIR_CCCL=/opt/cccl-src \
    && ninja -C cpp/build_cuda -j "$(nproc)"

# Native NVIDIA RAPIDS (cudf 24.12) + cuOpt python deps + REST server deps, baked into an
# isolated venv so the base conda/torch env stays untouched. cudf-cu12 comes from
# pypi.nvidia.com via the proxy; everything else from the Aliyun mirror direct (no_proxy),
# so only the RAPIDS wheels pay the proxy. This is the routing/distance + REST runtime for
# A100 (the numpy-only LP/MILP/QP path uses the same venv).
RUN /opt/conda/bin/python -m venv /opt/cuopt/.a100-venv \
    && http_proxy=${PROXY} https_proxy=${PROXY} \
       no_proxy=mirrors.aliyun.com,127.0.0.1,localhost \
       NO_PROXY=mirrors.aliyun.com,127.0.0.1,localhost \
       /opt/cuopt/.a100-venv/bin/pip install --no-cache-dir \
         -i http://mirrors.aliyun.com/pypi/simple/ \
         --extra-index-url https://pypi.nvidia.com \
         --trusted-host mirrors.aliyun.com \
         'cudf-cu12==24.12.*' cupy-cuda12x numba cython pytest numpy \
         'pandas>=2.0' 'pyyaml>=6.0.0' scipy networkx \
         fastapi 'uvicorn==0.34.*' 'msgpack==1.1.2' 'msgpack-numpy==0.4.8' \
         'jsonref==1.1.0' 'psutil>=6.0.0'

# Build the cuOpt Cython modules against the venv's numpy (ABI match for routing/REST),
# native NVIDIA path (g++ + libcudart).
RUN export PATH=/opt/cuopt/.a100-venv/bin:${PATH} \
    && CCCL=/opt/cccl-src LIBCUOPT_DIR=/opt/cuopt/cpp/build_cuda CUDA_HOME=/usr/local/cuda \
       bash python/cuopt/build_lp_modules_rapids_free.sh
