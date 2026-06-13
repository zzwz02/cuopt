# cmake-format: off
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
# cmake-format: on

# Use the CCCL surface shipped by MACA/cu-bridge instead of fetching NVIDIA
# CCCL.  cu-bridge provides newer cuda/* headers under include/cccl, while
# MACA's Thrust/CUB live in the regular cu-bridge and MACA include roots.
function(find_and_configure_maca_cccl)
  set(MACA_PATH "$ENV{MACA_PATH}" CACHE PATH "MACA toolkit root")
  if(NOT MACA_PATH)
    set(MACA_PATH "/opt/maca" CACHE PATH "MACA toolkit root" FORCE)
  endif()

  set(CUOPT_MACA_CUBRIDGE_INCLUDE_DIR
      "${MACA_PATH}/tools/cu-bridge/include"
      CACHE PATH "cu-bridge include directory")
  set(CUOPT_MACA_CUBRIDGE_CCCL_INCLUDE_DIR
      "${CUOPT_MACA_CUBRIDGE_INCLUDE_DIR}/cccl"
      CACHE PATH "cu-bridge CCCL compatibility include directory")
  set(CUOPT_MACA_INCLUDE_DIR
      "${MACA_PATH}/include"
      CACHE PATH "MACA toolkit include directory")

  foreach(_dir IN ITEMS
      "${CUOPT_MACA_CUBRIDGE_CCCL_INCLUDE_DIR}"
      "${CUOPT_MACA_CUBRIDGE_INCLUDE_DIR}"
      "${CUOPT_MACA_INCLUDE_DIR}")
    if(NOT EXISTS "${_dir}")
      message(FATAL_ERROR "cuOpt MACA CCCL include directory not found: ${_dir}")
    endif()
  endforeach()

  if(NOT TARGET CCCL::CCCL)
    add_library(CCCL::CCCL INTERFACE IMPORTED GLOBAL)
    set_target_properties(CCCL::CCCL PROPERTIES
      INTERFACE_INCLUDE_DIRECTORIES
        "${CUOPT_MACA_CUBRIDGE_CCCL_INCLUDE_DIR};${CUOPT_MACA_CUBRIDGE_INCLUDE_DIR};${CUOPT_MACA_INCLUDE_DIR}")
  endif()

  set(CCCL_SOURCE_DIR "" PARENT_SCOPE)
  message(STATUS "cuOpt: using MACA CCCL includes: ${CUOPT_MACA_CUBRIDGE_CCCL_INCLUDE_DIR};${CUOPT_MACA_CUBRIDGE_INCLUDE_DIR};${CUOPT_MACA_INCLUDE_DIR}")
endfunction()

find_and_configure_maca_cccl()
