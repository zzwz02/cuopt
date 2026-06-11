/*
 * cuOpt vendored RMM shim — typed uninitialized device vector over device_buffer.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <rmm/cuda_stream.hpp>       // transitively expected (owning streams)
#include <rmm/cuda_stream_pool.hpp>  // transitively expected (stream pools)
#include <rmm/cuda_stream_view.hpp>
#include <rmm/detail/error.hpp>
#include <rmm/device_buffer.hpp>
#include <rmm/mr/per_device_resource.hpp>
#include <rmm/resource_ref.hpp>

#include <cuda/std/iterator>

#include <cstddef>
#include <cstdint>

namespace rmm {

template <typename T>
class device_uvector {
  static_assert(std::is_trivially_copyable_v<T>,
                "device_uvector only supports types that are trivially copyable.");

 public:
  using value_type      = T;
  using size_type       = std::size_t;
  using reference       = value_type&;
  using const_reference = value_type const&;
  using pointer         = value_type*;
  using const_pointer   = value_type const*;
  using iterator        = pointer;
  using const_iterator  = const_pointer;

  ~device_uvector()                                = default;
  device_uvector(device_uvector&&) noexcept        = default;
  device_uvector& operator=(device_uvector&&) noexcept = default;
  device_uvector(device_uvector const&)            = delete;
  device_uvector& operator=(device_uvector const&) = delete;
  device_uvector()                                 = delete;

  explicit device_uvector(std::size_t size,
                          cuda_stream_view stream,
                          device_async_resource_ref mr = mr::get_current_device_resource_ref())
    : storage_{elements_to_bytes(size), stream, mr}
  {
  }

  device_uvector(device_uvector const& other,
                 cuda_stream_view stream,
                 device_async_resource_ref mr = mr::get_current_device_resource_ref())
    : storage_{other.storage_, stream, mr}
  {
  }

  [[nodiscard]] pointer element_ptr(size_type i) noexcept { return data() + i; }
  [[nodiscard]] const_pointer element_ptr(size_type i) const noexcept { return data() + i; }

  void set_element_async(size_type i, value_type const& value, cuda_stream_view stream)
  {
    RMM_EXPECTS(i < size(), "Attempt to access out of bounds element.", rmm::out_of_range);
    RMM_CUDA_TRY(
      cudaMemcpyAsync(element_ptr(i), &value, sizeof(value), cudaMemcpyDefault, stream.value()));
  }
  void set_element_async(size_type, value_type const&&, cuda_stream_view) = delete;

  void set_element_to_zero_async(size_type i, cuda_stream_view stream)
  {
    RMM_EXPECTS(i < size(), "Attempt to access out of bounds element.", rmm::out_of_range);
    RMM_CUDA_TRY(cudaMemsetAsync(element_ptr(i), 0, sizeof(value_type), stream.value()));
  }

  void set_element(size_type i, T const& value, cuda_stream_view stream)
  {
    set_element_async(i, value, stream);
    stream.synchronize();
  }

  [[nodiscard]] value_type element(size_type i, cuda_stream_view stream) const
  {
    RMM_EXPECTS(i < size(), "Attempt to access out of bounds element.", rmm::out_of_range);
    value_type value;
    RMM_CUDA_TRY(
      cudaMemcpyAsync(&value, element_ptr(i), sizeof(value), cudaMemcpyDefault, stream.value()));
    stream.synchronize();
    return value;
  }

  [[nodiscard]] value_type front_element(cuda_stream_view stream) const
  {
    return element(0, stream);
  }
  [[nodiscard]] value_type back_element(cuda_stream_view stream) const
  {
    return element(size() - 1, stream);
  }

  void reserve(std::size_t new_capacity, cuda_stream_view stream)
  {
    storage_.reserve(elements_to_bytes(new_capacity), stream);
  }

  void resize(std::size_t new_size, cuda_stream_view stream)
  {
    storage_.resize(elements_to_bytes(new_size), stream);
  }

  void shrink_to_fit(cuda_stream_view stream) { storage_.shrink_to_fit(stream); }

  [[nodiscard]] device_buffer release() noexcept { return std::move(storage_); }

  [[nodiscard]] std::size_t capacity() const noexcept
  {
    return bytes_to_elements(storage_.capacity());
  }

  [[nodiscard]] pointer data() noexcept { return static_cast<pointer>(storage_.data()); }
  [[nodiscard]] const_pointer data() const noexcept
  {
    return static_cast<const_pointer>(storage_.data());
  }

  [[nodiscard]] iterator begin() noexcept { return data(); }
  [[nodiscard]] const_iterator cbegin() const noexcept { return data(); }
  [[nodiscard]] const_iterator begin() const noexcept { return cbegin(); }

  [[nodiscard]] iterator end() noexcept { return data() + size(); }
  [[nodiscard]] const_iterator cend() const noexcept { return data() + size(); }
  [[nodiscard]] const_iterator end() const noexcept { return cend(); }

  [[nodiscard]] std::size_t size() const noexcept { return bytes_to_elements(storage_.size()); }
  [[nodiscard]] std::int64_t ssize() const noexcept { return static_cast<std::int64_t>(size()); }
  [[nodiscard]] bool is_empty() const noexcept { return size() == 0; }

  [[nodiscard]] device_async_resource_ref memory_resource() const noexcept
  {
    return storage_.memory_resource();
  }

  [[nodiscard]] cuda_stream_view stream() const noexcept { return storage_.stream(); }
  void set_stream(cuda_stream_view stream) noexcept { storage_.set_stream(stream); }

 private:
  device_buffer storage_;

  [[nodiscard]] static std::size_t elements_to_bytes(std::size_t num_elements) noexcept
  {
    return num_elements * sizeof(value_type);
  }
  [[nodiscard]] static std::size_t bytes_to_elements(std::size_t num_bytes) noexcept
  {
    return num_bytes / sizeof(value_type);
  }
};

}  // namespace rmm
