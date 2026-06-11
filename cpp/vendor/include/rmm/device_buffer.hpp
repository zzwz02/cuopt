/*
 * cuOpt vendored RMM shim — uninitialized stream-ordered device buffer.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <rmm/aligned.hpp>
#include <rmm/cuda_stream_view.hpp>
#include <rmm/detail/error.hpp>
#include <rmm/mr/device_memory_resource.hpp>
#include <rmm/mr/per_device_resource.hpp>
#include <rmm/resource_ref.hpp>

#include <cuda_runtime_api.h>

#include <cstddef>
#include <cstdint>
#include <utility>

namespace rmm {

/**
 * @brief Untyped, *uninitialized* stream-ordered device memory allocation.
 *
 * Allocation and free happen on the stream passed at construction / most
 * recently set via set_stream(). Contents are uninitialized (no fill).
 */
class device_buffer {
 public:
  device_buffer() : device_buffer(0, cuda_stream_view{}) {}

  explicit device_buffer(std::size_t size,
                         cuda_stream_view stream,
                         device_async_resource_ref mr = mr::get_current_device_resource_ref())
    : stream_{stream}, mr_{mr}
  {
    allocate_async(size);
  }

  device_buffer(void const* source_data,
                std::size_t size,
                cuda_stream_view stream,
                device_async_resource_ref mr = mr::get_current_device_resource_ref())
    : stream_{stream}, mr_{mr}
  {
    allocate_async(size);
    copy_async(source_data, size);
  }

  device_buffer(device_buffer const& other,
                cuda_stream_view stream,
                device_async_resource_ref mr = mr::get_current_device_resource_ref())
    : stream_{stream}, mr_{mr}
  {
    allocate_async(other.size());
    copy_async(other.data(), other.size());
  }

  device_buffer(device_buffer const&)            = delete;
  device_buffer& operator=(device_buffer const&) = delete;

  device_buffer(device_buffer&& other) noexcept
    : data_{other.data_},
      size_{other.size_},
      capacity_{other.capacity_},
      stream_{other.stream_},
      mr_{other.mr_}
  {
    other.data_     = nullptr;
    other.size_     = 0;
    other.capacity_ = 0;
    other.stream_   = cuda_stream_view{};
  }

  device_buffer& operator=(device_buffer&& other) noexcept
  {
    if (&other != this) {
      deallocate_async();
      data_     = other.data_;
      size_     = other.size_;
      capacity_ = other.capacity_;
      stream_   = other.stream_;
      mr_       = other.mr_;

      other.data_     = nullptr;
      other.size_     = 0;
      other.capacity_ = 0;
      other.stream_   = cuda_stream_view{};
    }
    return *this;
  }

  ~device_buffer() noexcept
  {
    deallocate_async();
    data_     = nullptr;
    size_     = 0;
    capacity_ = 0;
  }

  void reserve(std::size_t new_capacity, cuda_stream_view stream)
  {
    set_stream(stream);
    if (new_capacity > capacity_) {
      auto tmp        = device_buffer{new_capacity, stream, mr_};
      auto const size = size_;
      if (size != 0) {
        RMM_CUDA_TRY(cudaMemcpyAsync(tmp.data(), data(), size, cudaMemcpyDefault, stream.value()));
      }
      *this = std::move(tmp);
      size_ = size;
    }
  }

  void resize(std::size_t new_size, cuda_stream_view stream)
  {
    set_stream(stream);
    if (new_size <= capacity_) {
      size_ = new_size;
    } else {
      auto tmp        = device_buffer{new_size, stream, mr_};
      auto const size = size_;
      if (size != 0) {
        RMM_CUDA_TRY(cudaMemcpyAsync(tmp.data(), data(), size, cudaMemcpyDefault, stream.value()));
      }
      *this = std::move(tmp);
    }
  }

  void shrink_to_fit(cuda_stream_view stream)
  {
    set_stream(stream);
    if (size_ != capacity_) {
      auto tmp = device_buffer{size_, stream, mr_};
      if (size_ != 0) {
        RMM_CUDA_TRY(cudaMemcpyAsync(tmp.data(), data(), size_, cudaMemcpyDefault, stream.value()));
      }
      auto const size = size_;
      *this           = std::move(tmp);
      size_           = size;
    }
  }

  [[nodiscard]] void const* data() const noexcept { return data_; }
  [[nodiscard]] void* data() noexcept { return data_; }
  [[nodiscard]] std::size_t size() const noexcept { return size_; }
  [[nodiscard]] std::int64_t ssize() const noexcept { return static_cast<std::int64_t>(size_); }
  [[nodiscard]] bool is_empty() const noexcept { return size_ == 0; }
  [[nodiscard]] std::size_t capacity() const noexcept { return capacity_; }

  [[nodiscard]] cuda_stream_view stream() const noexcept { return stream_; }
  void set_stream(cuda_stream_view stream) noexcept { stream_ = stream; }

  [[nodiscard]] device_async_resource_ref memory_resource() const noexcept { return mr_; }

 private:
  void* data_{nullptr};
  std::size_t size_{0};
  std::size_t capacity_{0};
  cuda_stream_view stream_{};
  device_async_resource_ref mr_{mr::get_current_device_resource_ref()};

  void allocate_async(std::size_t bytes)
  {
    size_     = bytes;
    capacity_ = bytes;
    data_     = (bytes > 0) ? mr_.allocate_async(bytes, CUDA_ALLOCATION_ALIGNMENT, stream_)
                            : nullptr;
  }

  void deallocate_async() noexcept
  {
    if (capacity_ > 0 && data_ != nullptr) {
      mr_.deallocate_async(data_, capacity_, CUDA_ALLOCATION_ALIGNMENT, stream_);
    }
  }

  void copy_async(void const* source, std::size_t bytes)
  {
    if (bytes > 0) {
      RMM_EXPECTS(source != nullptr, "Invalid copy from nullptr.");
      RMM_EXPECTS(data_ != nullptr, "Invalid copy to nullptr.");
      RMM_CUDA_TRY(cudaMemcpyAsync(data_, source, bytes, cudaMemcpyDefault, stream_.value()));
    }
  }
};

}  // namespace rmm
