# cmake-format: off
# SPDX-FileCopyrightText: Copyright (c) 2021-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# cmake-format: on

# Fetch CCCL (thrust/cub/libcu++) via plain FetchContent. CCCL is an NVIDIA
# product that ships with the CUDA toolkit — not a RAPIDS library — but cuOpt
# pins a newer version than the toolkit bundles. Override the source with
# -DFETCHCONTENT_SOURCE_DIR_CCCL=<dir> for offline builds.
function(find_and_configure_cccl)
        include(FetchContent)
        FetchContent_Declare(
                CCCL
                GIT_REPOSITORY https://github.com/NVIDIA/cccl.git
                GIT_TAG v3.4.0
                GIT_SHALLOW TRUE
        )
        # CCCL's CMake provides the CCCL::CCCL interface target.
        FetchContent_MakeAvailable(CCCL)
        set(CCCL_SOURCE_DIR "${cccl_SOURCE_DIR}" PARENT_SCOPE)

        # nvcc < 12.4 (cudafe++) fails with "Internal Compiler Error (codegen):
        # internal error during structure layout!" on [[no_unique_address]]
        # members of class templates. Disable the attribute inside the fetched
        # CCCL for those toolchains. The disable must be unconditional (not
        # gated on the compiler doing the current pass) so every TU of the
        # build agrees on type layouts.
        if (CMAKE_CUDA_COMPILER_VERSION VERSION_LESS 12.4 AND DEFINED cccl_SOURCE_DIR)
                set(_cccl_attr_h "${cccl_SOURCE_DIR}/libcudacxx/include/cuda/std/__cccl/attributes.h")
                if (EXISTS "${_cccl_attr_h}")
                        file(READ "${_cccl_attr_h}" _cccl_attr_content)
                        set(_anchor "#endif // _CCCL_HAS_ATTRIBUTE_NO_UNIQUE_ADDRESS() && _CCCL_COMPILER(CLANG)")
                        set(_nvcc_workaround
"${_anchor}

// cuOpt workaround: nvcc < 12.4 crashes (\"internal error during structure
// layout\") on [[no_unique_address]] members of class templates. Disabled for
// all compilers so layouts stay consistent across gcc/nvcc TUs.
#undef _CCCL_HAS_ATTRIBUTE_NO_UNIQUE_ADDRESS
#undef _CCCL_NO_UNIQUE_ADDRESS
#define _CCCL_HAS_ATTRIBUTE_NO_UNIQUE_ADDRESS() 0
#define _CCCL_NO_UNIQUE_ADDRESS")
                        if (NOT _cccl_attr_content MATCHES "cuOpt workaround")
                                string(REPLACE "${_anchor}" "${_nvcc_workaround}" _cccl_attr_content "${_cccl_attr_content}")
                                file(WRITE "${_cccl_attr_h}" "${_cccl_attr_content}")
                                message(STATUS "cuOpt: patched CCCL to disable [[no_unique_address]] for nvcc ${CMAKE_CUDA_COMPILER_VERSION}")
                        endif ()
                endif ()
        endif ()
endfunction()

find_and_configure_cccl()
