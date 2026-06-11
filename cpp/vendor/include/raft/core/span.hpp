/*
 * cuOpt vendored RAFT shim — span over cuda::std::span (host or device).
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <cuda/std/cstddef>
#include <cuda/std/span>

#include <cstddef>
#include <type_traits>

namespace raft {

inline constexpr std::size_t dynamic_extent = cuda::std::dynamic_extent;

/**
 * @brief raft span: a non-owning view with raft's template signature
 * `span<T, is_device, Extent>`, implemented over cuda::std::span.
 */
template <typename T, bool is_device, std::size_t Extent = dynamic_extent>
class span {
  using base_type = cuda::std::span<T, Extent>;

 public:
  using element_type    = T;
  using value_type      = std::remove_cv_t<T>;
  using size_type       = std::size_t;
  using pointer         = T*;
  using const_pointer   = T const*;
  using reference       = T&;
  using const_reference = T const&;
  using iterator        = typename base_type::iterator;

  static constexpr bool is_device_span = is_device;

  __host__ __device__ constexpr span() noexcept = default;

  __host__ __device__ constexpr span(pointer ptr, size_type count) noexcept : base_{ptr, count} {}

  __host__ __device__ constexpr span(pointer first, pointer last) noexcept
    : base_{first, static_cast<size_type>(last - first)}
  {
  }

  template <std::size_t N>
  __host__ __device__ constexpr span(element_type (&arr)[N]) noexcept : base_{arr, N}
  {
  }

  __host__ __device__ constexpr span(base_type other) noexcept : base_{other} {}

  __host__ __device__ constexpr span(span const&) noexcept            = default;
  __host__ __device__ constexpr span& operator=(span const&) noexcept = default;

  __host__ __device__ constexpr pointer data() const noexcept { return base_.data(); }
  __host__ __device__ constexpr size_type size() const noexcept { return base_.size(); }
  __host__ __device__ constexpr size_type size_bytes() const noexcept
  {
    return base_.size() * sizeof(T);
  }
  __host__ __device__ constexpr bool empty() const noexcept { return base_.empty(); }

  __host__ __device__ constexpr reference operator[](size_type i) const noexcept { return base_[i]; }
  __host__ __device__ constexpr reference front() const noexcept { return base_.front(); }
  __host__ __device__ constexpr reference back() const noexcept { return base_.back(); }

  __host__ __device__ constexpr iterator begin() const noexcept { return base_.begin(); }
  __host__ __device__ constexpr iterator end() const noexcept { return base_.end(); }

  __host__ __device__ constexpr span<T, is_device, dynamic_extent> subspan(
    size_type offset, size_type count = dynamic_extent) const noexcept
  {
    return span<T, is_device, dynamic_extent>{
      data() + offset, count == dynamic_extent ? size() - offset : count};
  }

 private:
  base_type base_{};
};

}  // namespace raft
