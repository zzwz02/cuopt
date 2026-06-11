/*
 * cuOpt vendored RMM shim — non-owning resource reference (cuda::mr-free).
 * Replaces rmm::device_async_resource_ref = cuda::mr::async_resource_ref<...>.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <rmm/cuda_stream_view.hpp>
#include <rmm/mr/device_memory_resource.hpp>

#include <cstddef>

namespace rmm {

/**
 * @brief Lightweight non-owning reference to a device_memory_resource.
 *
 * Mirrors the call surface of rmm::device_async_resource_ref without depending
 * on cuda::mr.
 */
class device_async_resource_ref {
 public:
  device_async_resource_ref() = default;

  // NOLINTNEXTLINE(google-explicit-constructor)
  device_async_resource_ref(mr::device_memory_resource& mr) noexcept : mr_{&mr} {}
  // NOLINTNEXTLINE(google-explicit-constructor)
  device_async_resource_ref(mr::device_memory_resource* mr) noexcept : mr_{mr} {}

  [[nodiscard]] void* allocate_async(std::size_t bytes,
                                     std::size_t alignment,
                                     cuda_stream_view stream)
  {
    return mr_->allocate_async(bytes, alignment, stream);
  }
  [[nodiscard]] void* allocate_async(std::size_t bytes, cuda_stream_view stream)
  {
    return mr_->allocate_async(bytes, stream);
  }
  void deallocate_async(void* ptr,
                        std::size_t bytes,
                        std::size_t alignment,
                        cuda_stream_view stream)
  {
    mr_->deallocate_async(ptr, bytes, alignment, stream);
  }
  void deallocate_async(void* ptr, std::size_t bytes, cuda_stream_view stream)
  {
    mr_->deallocate_async(ptr, bytes, stream);
  }

  [[nodiscard]] mr::device_memory_resource* get() const noexcept { return mr_; }

  bool operator==(device_async_resource_ref const& other) const noexcept
  {
    return mr_ == other.mr_ || (mr_ != nullptr && other.mr_ != nullptr && mr_->is_equal(*other.mr_));
  }
  bool operator!=(device_async_resource_ref const& other) const noexcept
  {
    return !(*this == other);
  }

 private:
  mr::device_memory_resource* mr_{nullptr};
};

using device_resource_ref               = device_async_resource_ref;
using host_device_async_resource_ref    = device_async_resource_ref;

}  // namespace rmm
