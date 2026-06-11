/*
 * cuOpt vendored RMM shim — Thrust allocator backed by a device resource.
 * cuda::mr-free. SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <rmm/aligned.hpp>
#include <rmm/cuda_device.hpp>
#include <rmm/cuda_stream_view.hpp>
#include <rmm/mr/per_device_resource.hpp>
#include <rmm/resource_ref.hpp>

#include <thrust/device_malloc_allocator.h>
#include <thrust/device_ptr.h>

#include <cstddef>
#include <utility>

namespace rmm::mr {

/**
 * @brief Thrust allocator that draws temporary storage from a device resource
 * on a given stream (the same pool used by device_uvector).
 */
template <typename T>
class thrust_allocator : public thrust::device_malloc_allocator<T> {
 public:
  using Base      = thrust::device_malloc_allocator<T>;
  using pointer   = typename Base::pointer;
  using size_type = typename Base::size_type;

  template <typename U>
  struct rebind {
    using other = thrust_allocator<U>;
  };

  thrust_allocator() = default;

  explicit thrust_allocator(cuda_stream_view stream) : stream_{stream} {}

  thrust_allocator(cuda_stream_view stream, device_async_resource_ref mr)
    : stream_{stream}, mr_{mr}
  {
  }

  template <typename U>
  thrust_allocator(thrust_allocator<U> const& other)
    : mr_{other.get_upstream_resource()}, stream_{other.stream()}, device_{other.device()}
  {
  }

  pointer allocate(size_type num)
  {
    cuda_set_device_raii dev{device_};
    return thrust::device_pointer_cast(static_cast<T*>(
      mr_.allocate_async(num * sizeof(T), rmm::CUDA_ALLOCATION_ALIGNMENT, stream_)));
  }

  void deallocate(pointer ptr, size_type num) noexcept
  {
    cuda_set_device_raii dev{device_};
    mr_.deallocate_async(
      thrust::raw_pointer_cast(ptr), num * sizeof(T), rmm::CUDA_ALLOCATION_ALIGNMENT, stream_);
  }

  [[nodiscard]] device_async_resource_ref get_upstream_resource() const noexcept { return mr_; }
  [[nodiscard]] cuda_stream_view stream() const noexcept { return stream_; }
  [[nodiscard]] cuda_device_id device() const noexcept { return device_; }

 private:
  cuda_stream_view stream_{};
  device_async_resource_ref mr_{get_current_device_resource_ref()};
  cuda_device_id device_{get_current_cuda_device()};
};

}  // namespace rmm::mr
