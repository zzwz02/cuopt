/*
 * cuOpt vendored RMM shim — stream-ordered pool resource over cudaMallocAsync.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <rmm/cuda_device.hpp>
#include <rmm/cuda_stream_view.hpp>
#include <rmm/detail/error.hpp>
#include <rmm/mr/device_memory_resource.hpp>

#include <cuda_runtime_api.h>

#include <cstddef>
#include <optional>

namespace rmm::mr {

/**
 * @brief Device memory resource backed by a CUDA stream-ordered memory pool.
 *
 * Uses cudaMallocFromPoolAsync / cudaFreeAsync. By default it adopts the
 * device's default mempool with an effectively-unbounded release threshold so
 * freed memory is retained for reuse (matching rmm pool semantics).
 */
class cuda_async_memory_resource final : public device_memory_resource {
 public:
  struct pool_props {
    std::optional<std::size_t> initial_pool_size{};
    std::optional<std::size_t> release_threshold{};
  };

  cuda_async_memory_resource() : cuda_async_memory_resource(pool_props{}) {}

  explicit cuda_async_memory_resource(std::optional<std::size_t> initial_pool_size,
                                      std::optional<std::size_t> release_threshold = {})
    : cuda_async_memory_resource(pool_props{initial_pool_size, release_threshold})
  {
  }

  explicit cuda_async_memory_resource(pool_props props)
    : device_id_{get_current_cuda_device()}
  {
    int supported{0};
    RMM_CUDA_TRY(cudaDeviceGetAttribute(
      &supported, cudaDevAttrMemoryPoolsSupported, device_id_.value()));
    RMM_EXPECTS(supported != 0, "cudaMallocAsync not supported on this device");
    RMM_CUDA_TRY(cudaDeviceGetDefaultMemPool(&pool_, device_id_.value()));

    std::uint64_t const threshold =
      props.release_threshold.value_or(static_cast<std::uint64_t>(UINT64_MAX));
    RMM_CUDA_TRY(
      cudaMemPoolSetAttribute(pool_, cudaMemPoolAttrReleaseThreshold, (void*)&threshold));

    if (props.initial_pool_size.has_value() && *props.initial_pool_size > 0) {
      // Prime the pool by allocating and immediately freeing initial_pool_size.
      cuda_set_device_raii set_dev{device_id_};
      void* ptr{nullptr};
      RMM_CUDA_TRY(cudaMallocFromPoolAsync(&ptr, *props.initial_pool_size, pool_, cudaStreamLegacy));
      RMM_CUDA_TRY(cudaFreeAsync(ptr, cudaStreamLegacy));
      RMM_CUDA_TRY(cudaStreamSynchronize(cudaStreamLegacy));
    }
  }

  [[nodiscard]] cudaMemPool_t pool_handle() const noexcept { return pool_; }

 private:
  void* do_allocate(std::size_t bytes, cuda_stream_view stream) override
  {
    void* ptr{nullptr};
    if (bytes == 0) { return nullptr; }
    cuda_set_device_raii set_dev{device_id_};
    RMM_CUDA_TRY_ALLOC(cudaMallocFromPoolAsync(&ptr, bytes, pool_, stream.value()), bytes);
    return ptr;
  }

  void do_deallocate(void* ptr, std::size_t, cuda_stream_view stream) override
  {
    if (ptr == nullptr) { return; }
    cuda_set_device_raii set_dev{device_id_};
    auto const status = cudaFreeAsync(ptr, stream.value());
    if (status != cudaSuccess && status != cudaErrorCudartUnloading) { cudaGetLastError(); }
  }

  [[nodiscard]] bool do_is_equal(device_memory_resource const& other) const noexcept override
  {
    auto const* cast = dynamic_cast<cuda_async_memory_resource const*>(&other);
    return cast != nullptr && cast->pool_ == pool_;
  }

  cuda_device_id device_id_;
  cudaMemPool_t pool_{};
};

}  // namespace rmm::mr
