/*
 * cuOpt vendored RMM shim — cudaMallocManaged resource.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <rmm/cuda_stream_view.hpp>
#include <rmm/detail/error.hpp>
#include <rmm/mr/device_memory_resource.hpp>

#include <cuda_runtime_api.h>

namespace rmm::mr {

/** @brief Device memory resource using CUDA managed (unified) memory. */
class managed_memory_resource final : public device_memory_resource {
 private:
  void* do_allocate(std::size_t bytes, cuda_stream_view) override
  {
    void* ptr{nullptr};
    if (bytes == 0) { return nullptr; }
    RMM_CUDA_TRY_ALLOC(cudaMallocManaged(&ptr, bytes), bytes);
    return ptr;
  }

  void do_deallocate(void* ptr, std::size_t, cuda_stream_view) override
  {
    if (ptr == nullptr) { return; }
    auto const status = cudaFree(ptr);
    if (status != cudaSuccess && status != cudaErrorCudartUnloading) { cudaGetLastError(); }
  }

  [[nodiscard]] bool do_is_equal(device_memory_resource const& other) const noexcept override
  {
    return dynamic_cast<managed_memory_resource const*>(&other) != nullptr;
  }
};

}  // namespace rmm::mr
