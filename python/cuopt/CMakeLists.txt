# cmake-format: off
# SPDX-FileCopyrightText: Copyright (c) 2023-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# cmake-format: on

cmake_minimum_required(VERSION 3.30 FATAL_ERROR)

# Plain-CMake build (RAPIDS-free): no rapids-cmake / rapids-cython-core.
file(READ "${CMAKE_CURRENT_SOURCE_DIR}/cuopt/VERSION" _cuopt_version_raw)
string(STRIP "${_cuopt_version_raw}" CUOPT_PY_VERSION)

project(
  cuopt-python
  VERSION "${CUOPT_PY_VERSION}"
  LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# libcuopt: point CMAKE_PREFIX_PATH at an installed libcuopt (wheel layout:
# <site-packages>/libcuopt) or at a cpp build tree (cpp/build exports a
# build-tree cuopt-config.cmake).
find_package(cuopt "${CUOPT_PY_VERSION}" REQUIRED)

find_package(Python3 REQUIRED COMPONENTS Interpreter Development.Module)

# Plain-CMake replacement for rapids_cython_create_modules so the per-module
# CMakeLists.txt files stay unchanged. For each .pyx: cythonize (C++ mode,
# package root on the cimport path), build a Python extension module, link the
# requested libraries, and install it next to its source package with an
# RPATH that reaches the libcuopt wheel's lib64 from that depth.
function(rapids_cython_create_modules)
  set(_options CXX)
  set(_one_value MODULE_PREFIX)
  set(_multi_value SOURCE_FILES LINKED_LIBRARIES ASSOCIATED_TARGETS)
  cmake_parse_arguments(RCM "${_options}" "${_one_value}" "${_multi_value}" ${ARGN})
  foreach(src IN LISTS RCM_SOURCE_FILES)
    get_filename_component(mod "${src}" NAME_WE)
    set(tgt "${RCM_MODULE_PREFIX}${mod}")
    set(gen "${CMAKE_CURRENT_BINARY_DIR}/${tgt}_cython.cpp")
    add_custom_command(
      OUTPUT "${gen}"
      COMMAND "${Python3_EXECUTABLE}" -m cython --cplus -3 -I
              "${CMAKE_SOURCE_DIR}" "${CMAKE_CURRENT_SOURCE_DIR}/${src}" -o
              "${gen}"
      DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/${src}"
      VERBATIM
      COMMENT "Cythonizing ${src}")
    python3_add_library("${tgt}" MODULE WITH_SOABI "${gen}")
    set_target_properties("${tgt}" PROPERTIES OUTPUT_NAME "${mod}")
    if(RCM_LINKED_LIBRARIES)
      target_link_libraries("${tgt}" PRIVATE ${RCM_LINKED_LIBRARIES})
    endif()
    file(RELATIVE_PATH _rel "${CMAKE_SOURCE_DIR}" "${CMAKE_CURRENT_SOURCE_DIR}")
    string(REGEX REPLACE "[^/]+" ".." _ups "${_rel}")
    set_property(
      TARGET "${tgt}"
      PROPERTY INSTALL_RPATH "$ORIGIN/${_ups}/libcuopt/lib64"
               "$ORIGIN/${_ups}/libcuopt/lib")
    install(TARGETS "${tgt}" DESTINATION "${_rel}")
  endforeach()
endfunction()

add_subdirectory(cuopt)
add_subdirectory(cuopt/linear_programming/internals/)
