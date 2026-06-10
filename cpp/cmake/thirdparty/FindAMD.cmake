# cmake-format: off
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# cmake-format: on

# FindAMD.cmake - Find the SuiteSparse AMD (approximate minimum degree) library
#
# Used by the barrier solver's built-in sparse LDL^T factorization as the
# fill-reducing ordering. Provided by e.g. libsuitesparse-dev (apt) or
# suitesparse (conda-forge).
#
# This module defines the following variables:
#   AMD_FOUND        - True if AMD is found
#   AMD_INCLUDE_DIRS - AMD include directory (containing amd.h)
#   AMD_LIBRARIES    - AMD library
#   SuiteSparse::AMD - Imported target

find_path(AMD_INCLUDE_DIR
  NAMES amd.h
  PATH_SUFFIXES suitesparse
  PATHS
    /usr/include
    /usr/local/include
    /opt/conda/include
    $ENV{CONDA_PREFIX}/include
)

find_library(AMD_LIBRARY
  NAMES amd
  PATHS
    /usr/lib
    /usr/lib/x86_64-linux-gnu
    /usr/lib/aarch64-linux-gnu
    /usr/local/lib
    /opt/conda/lib
    $ENV{CONDA_PREFIX}/lib
)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(AMD
  REQUIRED_VARS AMD_LIBRARY AMD_INCLUDE_DIR
)

if(AMD_FOUND)
  set(AMD_LIBRARIES ${AMD_LIBRARY})
  set(AMD_INCLUDE_DIRS ${AMD_INCLUDE_DIR})

  if(NOT TARGET SuiteSparse::AMD)
    add_library(SuiteSparse::AMD UNKNOWN IMPORTED)
    set_target_properties(SuiteSparse::AMD PROPERTIES
      IMPORTED_LOCATION "${AMD_LIBRARY}"
      INTERFACE_INCLUDE_DIRECTORIES "${AMD_INCLUDE_DIR}"
    )
  endif()
endif()

mark_as_advanced(AMD_INCLUDE_DIR AMD_LIBRARY)
