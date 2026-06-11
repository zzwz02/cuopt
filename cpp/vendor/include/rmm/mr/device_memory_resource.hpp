/*
 * cuOpt vendored RMM shim — abstract device memory resource (cuda::mr-free).
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <rmm/aligned.hpp>
#include <rmm/cuda_stream_view.hpp>
#include <rmm/detail/error.hpp>

#include <cstddef>

namespace rmm::mr {

/**
 * @brief Base class for all stream-ordered device memory resources.
 *
 * This replaces RAFT/RMM's cuda::mr-concept-based hierarchy with a plain
 * virtual interface so the shim carries no libcu++ memory-resource dependency.
 */
class device_memory_resource {
 public:
  device_memory_resource()                                         = default;
  virtual ~device_memory_resource()                               = default;
  device_memory_resource(device_memory_resource const&)            = default;
  device_memory_resource& operator=(device_memory_resource const&) = default;
  device_memory_resource(device_memory_resource&&) noexcept        = default;
  device_memory_resource& operator=(device_memory_resource&&) noexcept = default;

  void* allocate(std::size_t bytes, cuda_stream_view stream = cuda_stream_view{})
  {
    return do_allocate(align_up(bytes, CUDA_ALLOCATION_ALIGNMENT), stream);
  }

  void deallocate(void* ptr, std::size_t bytes, cuda_stream_view stream = cuda_stream_view{})
  {
    do_deallocate(ptr, align_up(bytes, CUDA_ALLOCATION_ALIGNMENT), stream);
  }

  // cuda::mr async-resource style aliases used by device_async_resource_ref.
  void* allocate_async(std::size_t bytes, std::size_t alignment, cuda_stream_view stream)
  {
    return do_allocate(align_up(bytes, alignment), stream);
  }
  void* allocate_async(std::size_t bytes, cuda_stream_view stream)
  {
    return allocate(bytes, stream);
  }
  void deallocate_async(void* ptr,
                        std::size_t bytes,
                        std::size_t alignment,
                        cuda_stream_view stream)
  {
    do_deallocate(ptr, align_up(bytes, alignment), stream);
  }
  void deallocate_async(void* ptr, std::size_t bytes, cuda_stream_view stream)
  {
    deallocate(ptr, bytes, stream);
  }

  [[nodiscard]] bool is_equal(device_memory_resource const& other) const noexcept
  {
    return do_is_equal(other);
  }

  bool operator==(device_memory_resource const& other) const noexcept { return do_is_equal(other); }
  bool operator!=(device_memory_resource const& other) const noexcept
  {
    return !do_is_equal(other);
  }

 private:
  virtual void* do_allocate(std::size_t bytes, cuda_stream_view stream)                = 0;
  virtual void do_deallocate(void* ptr, std::size_t bytes, cuda_stream_view stream)    = 0;
  [[nodiscard]] virtual bool do_is_equal(device_memory_resource const& other) const noexcept
  {
    return this == &other;
  }
};

}  // namespace rmm::mr
