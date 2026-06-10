# cmake-format: off
# SPDX-FileCopyrightText: Copyright (c) 2021-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# cmake-format: on

set(CUOPT_MIN_VERSION_raft "${RAPIDS_VERSION_MAJOR}.${RAPIDS_VERSION_MINOR}.00")

function(find_and_configure_raft)
    set(oneValueArgs VERSION FORK PINNED_TAG CLONE_ON_PIN)
    cmake_parse_arguments(PKG "" "${oneValueArgs}" "" ${ARGN})

    if(PKG_CLONE_ON_PIN AND NOT PKG_PINNED_TAG STREQUAL "${rapids-cmake-checkout-tag}")
        message("Pinned tag found: ${PKG_PINNED_TAG}. Cloning raft locally.")
        set(CPM_DOWNLOAD_raft ON)
    endif()

    rapids_cpm_find(raft ${PKG_VERSION}
        GLOBAL_TARGETS raft::raft
        BUILD_EXPORT_SET cuopt-exports
        INSTALL_EXPORT_SET cuopt-exports
        CPM_ARGS
        GIT_REPOSITORY https://github.com/${PKG_FORK}/raft.git
        GIT_TAG ${PKG_PINNED_TAG}
        SOURCE_SUBDIR cpp
        OPTIONS
            "BUILD_TESTS OFF"
            "BUILD_PRIMS_BENCH OFF"
            "RAFT_COMPILE_LIBRARY OFF"
    )

    if(raft_ADDED)
        message(VERBOSE "CUOPT: Using RAFT located in ${raft_SOURCE_DIR}")
    else()
        message(VERBOSE "CUOPT: Using RAFT located in ${raft_DIR}")
    endif()

    # nvcc < 12.4 (cudafe++) miscompiles the brace default-member-initializers
    # in raft's nvtx_range_stack.hpp ("'current_' was not declared in this
    # scope"). Rewrite them into an explicit default constructor.
    if (CMAKE_CUDA_COMPILER_VERSION VERSION_LESS 12.4 AND DEFINED raft_SOURCE_DIR)
        set(_raft_nvtx_h "${raft_SOURCE_DIR}/cpp/include/raft/core/detail/nvtx_range_stack.hpp")
        if (EXISTS "${_raft_nvtx_h}")
            file(READ "${_raft_nvtx_h}" _raft_nvtx_content)
            if (NOT _raft_nvtx_content MATCHES "cuOpt workaround")
                string(REPLACE
"struct nvtx_range_name_stack {"
"struct nvtx_range_name_stack {
  // cuOpt workaround: explicit constructor instead of default member
  // initializers, which old cudafe++ front ends mis-parse.
  nvtx_range_name_stack() : stack_(), current_(std::make_shared<current_range>()) {}
"
                       _raft_nvtx_content "${_raft_nvtx_content}")
                string(REPLACE
"  std::stack<std::string> stack_{};
  std::shared_ptr<current_range> current_{std::make_shared<current_range>()};"
"  std::stack<std::string> stack_;
  std::shared_ptr<current_range> current_;"
                       _raft_nvtx_content "${_raft_nvtx_content}")
                file(WRITE "${_raft_nvtx_h}" "${_raft_nvtx_content}")
                message(STATUS "cuOpt: patched raft nvtx_range_stack.hpp for nvcc ${CMAKE_CUDA_COMPILER_VERSION}")
            endif ()
        endif ()
    endif ()
endfunction()

# Change pinned tag and fork here to test a commit in CI
# To use a different RAFT locally, set the CMake variable
# CPM_raft_SOURCE=/path/to/local/raft
find_and_configure_raft(VERSION ${CUOPT_MIN_VERSION_raft}
    FORK rapidsai
    PINNED_TAG ${rapids-cmake-checkout-tag}
    CLONE_ON_PIN ON
)
