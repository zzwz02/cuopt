/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#pragma once

#include <utilities/error.hpp>
#include <utilities/macros.cuh>
#include "cxxopts.hpp"

#include <gtest/gtest.h>

#include <rmm/mr/cuda_async_memory_resource.hpp>
#include <rmm/mr/cuda_memory_resource.hpp>
#include <rmm/mr/device_memory_resource.hpp>
#include <rmm/mr/managed_memory_resource.hpp>
#include <rmm/mr/per_device_resource.hpp>

#include <memory>

namespace cuopt {
namespace test {

using mr_ptr = std::shared_ptr<rmm::mr::device_memory_resource>;

/// MR factory functions (cuda::mr-free shim: owning shared_ptr resources).
inline mr_ptr make_cuda() { return std::make_shared<rmm::mr::cuda_memory_resource>(); }

inline mr_ptr make_async() { return std::make_shared<rmm::mr::cuda_async_memory_resource>(); }

inline mr_ptr make_managed() { return std::make_shared<rmm::mr::managed_memory_resource>(); }

inline mr_ptr make_pool()
{
  // 1GB of initial pool size, backed by the stream-ordered CUDA mempool.
  const size_t initial_pool_size = 1024 * 1024 * 1024;
  return std::make_shared<rmm::mr::cuda_async_memory_resource>(initial_pool_size);
}

// The mempool already pools allocations; "binning" maps to the pool resource.
inline mr_ptr make_binning() { return make_pool(); }

/**
 * @brief Creates a memory resource for the unit test environment given the name
 * of the allocation mode. The returned resource must be kept alive for the
 * duration of the tests.
 *
 * @param allocation_mode One of "binning", "cuda", "pool", "managed".
 * @return Owning memory resource handle.
 */
inline mr_ptr create_memory_resource(std::string const& allocation_mode)
{
  if (allocation_mode == "binning") return make_binning();
  if (allocation_mode == "cuda") return make_cuda();
  if (allocation_mode == "pool") return make_pool();
  if (allocation_mode == "managed") return make_managed();
  cuopt_assert(false, "Invalid RMM allocation mode");

  // control will never reach this point
  return make_managed();
}

}  // namespace test
}  // namespace cuopt

/**
 * @brief Parses the cuOpt test command line options.
 *
 * Currently only supports 'rmm_mode' string paramater, which set the rmm
 * allocation mode. The default value of the parameter is 'pool'.
 *
 * @return Parsing results in the form of cxxopts::ParseResult
 */
inline auto parse_test_options(int argc, char** argv)
{
  try {
    cxxopts::Options options(argv[0], " - cuOpt tests command line options");
    // Default to the stream-ordered async pool: cuOpt captures solver work into
    // CUDA graphs (manual_cuda_graph_t), and CUDA stream capture forbids the
    // synchronous cudaMalloc used by the plain "cuda" resource — only the pool's
    // cudaMallocFromPoolAsync is capturable.
    options.allow_unrecognised_options().add_options()(
      "rmm_mode", "RMM allocation mode", cxxopts::value<std::string>()->default_value("pool"));

    return options.parse(argc, argv);
  } catch (const std::exception& e) {
    cuopt_assert(false, "Error parsing command line options");
  }

  // control will never reach this point
  cxxopts::Options options(argv[0], " - cuOpt tests command line options");
  return options.parse(argc, argv);
}

/**
 * @brief Macro that defines main function for gtest programs that use rmm
 *
 * Should be included in every test program that uses rmm allocators since it
 * maintains the lifespan of the rmm default memory resource. This `main`
 * function is a wrapper around the google test generated `main`, maintaining
 * the original functionality. In addition, this custom `main` function parses
 * the command line to customize test behavior, like the allocation mode used
 * for creating the default memory resource.
 */
#define CUOPT_TEST_PROGRAM_MAIN()                                        \
  int main(int argc, char** argv)                                        \
  {                                                                      \
    ::testing::InitGoogleTest(&argc, argv);                              \
    auto const cmd_opts = parse_test_options(argc, argv);                \
    auto const rmm_mode = cmd_opts["rmm_mode"].as<std::string>();        \
    auto resource       = cuopt::test::create_memory_resource(rmm_mode); \
    rmm::mr::set_current_device_resource(resource.get());                \
    return RUN_ALL_TESTS();                                              \
  }
