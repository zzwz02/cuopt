# cmake-format: off
# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# cmake-format: on

cmake_minimum_required(VERSION 3.30 FATAL_ERROR)

# Plain-CMake build (RAPIDS-free): no rapids-cmake / rapids-cuda.
file(READ "${CMAKE_CURRENT_SOURCE_DIR}/libcuopt/VERSION" _libcuopt_version_raw)
string(STRIP "${_libcuopt_version_raw}" LIBCUOPT_PY_VERSION)

if(NOT CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES native)
endif()

project(
  libcuopt-python
  VERSION "${LIBCUOPT_PY_VERSION}"
  LANGUAGES CXX CUDA
)

# Debug: Print CUDA compiler version early
message(STATUS "=============================================================")
message(STATUS "libcuopt: CMAKE_CUDA_COMPILER_VERSION = ${CMAKE_CUDA_COMPILER_VERSION}")
message(STATUS "libcuopt: CUDAToolkit_VERSION = ${CUDAToolkit_VERSION}")
message(STATUS "=============================================================")

# Check if cuopt is already available. If so, it is the user's responsibility to ensure that the
# CMake package is also available at build time of the Python cuopt package.
find_package(cuopt "${LIBCUOPT_PY_VERSION}")

if(cuopt_FOUND)
  return()
endif()

unset(cuopt_FOUND)

include(FetchContent)
FetchContent_Declare(
  argparse
  GIT_REPOSITORY https://github.com/p-ranav/argparse.git
  GIT_TAG v3.2
)
FetchContent_MakeAvailable(argparse)

# gRPC must be available as an installed CMake package (gRPCConfig.cmake)
# unless the gRPC server is skipped (-DSKIP_GRPC_BUILD=ON).
if(NOT SKIP_GRPC_BUILD)
  find_package(gRPC CONFIG REQUIRED)
endif()

find_package(Boost 1.65 REQUIRED)
if(Boost_FOUND)
    message(STATUS "Found Boost ${Boost_VERSION} in ${Boost_INCLUDE_DIRS}")
else()
    message(FATAL_ERROR "Boost not found. Please install boost-devel.")
endif()

include_directories(${Boost_INCLUDE_DIRS})

set(BUILD_TESTING OFF CACHE BOOL "Disable test build for papilo")
set(PAPILO_NO_BINARIES ON)
option(LUSOL "Disable LUSOL" OFF)

set(BUILD_TESTS OFF)
set(BUILD_BENCHMARKS OFF)
set(CUOPT_BUILD_TESTUTIL OFF)

add_subdirectory(../../cpp cuopt-cpp)

target_link_libraries(cuopt PRIVATE
    argparse
)
target_link_libraries(cuopt_cli PRIVATE
    argparse
)

set(rpaths
  "$ORIGIN/../lib64"
  "$ORIGIN/../../nvidia/cublas/lib"
  "$ORIGIN/../../nvidia/curand/lib"
  "$ORIGIN/../../nvidia/cusolver/lib"
  "$ORIGIN/../../nvidia/cusparse/lib"
  "$ORIGIN/../../nvidia/nvjitlink/lib"
)

# Add CUDA version-specific paths based on CUDA compiler version
message(STATUS "libcuopt: CMAKE_CUDA_COMPILER_VERSION = ${CMAKE_CUDA_COMPILER_VERSION}")
if(CMAKE_CUDA_COMPILER_VERSION VERSION_GREATER_EQUAL 13.0 AND CMAKE_CUDA_COMPILER_VERSION VERSION_LESS 14.0)
  message(STATUS "libcuopt: Adding cu13 RPATH")
  list(APPEND rpaths "$ORIGIN/../../nvidia/cu13/lib")
elseif(CMAKE_CUDA_COMPILER_VERSION VERSION_GREATER_EQUAL 12.0 AND CMAKE_CUDA_COMPILER_VERSION VERSION_LESS 13.0)
  message(STATUS "libcuopt: Adding cu12 RPATH")
  list(APPEND rpaths "$ORIGIN/../../nvidia/cu12/lib")
else()
  message(WARNING "libcuopt: Unsupported CUDA version ${CMAKE_CUDA_COMPILER_VERSION}")
endif()
message(STATUS "libcuopt: Final RPATH = ${rpaths}")

set_property(TARGET cuopt PROPERTY INSTALL_RPATH ${rpaths} APPEND)
set_property(TARGET cuopt_cli PROPERTY INSTALL_RPATH ${rpaths} APPEND)
if(TARGET cuopt_grpc_server)
  set_property(TARGET cuopt_grpc_server PROPERTY INSTALL_RPATH ${rpaths} APPEND)
endif()

# Bundle the TBB runtime next to libcuopt.so when TBB is an external shared
# library (production wheels normally achieve this via auditwheel repair).
if(TBB_LIBRARY)
  get_filename_component(_tbb_real "${TBB_LIBRARY}" REALPATH)
  get_filename_component(_tbb_dir "${_tbb_real}" DIRECTORY)
  file(GLOB _tbb_runtime "${_tbb_dir}/libtbb.so*")
  if(_tbb_runtime)
    install(FILES ${_tbb_runtime} DESTINATION ${CMAKE_INSTALL_LIBDIR})
  endif()
endif()
