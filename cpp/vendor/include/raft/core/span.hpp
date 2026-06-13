/*
 * cuOpt vendored RAFT shim — span over cuda::std::span (host or device).
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <raft/core/detail/macros.hpp>

#include <cuda/std/cstddef>
#include <cuda/std/span>

#include <cstddef>
#include <type_traits>

namespace raft {

using cuda::std::dynamic_extent;

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
  // raft::span exposes raw-pointer iterators (cuOpt passes span.begin()/end()
  // to APIs expecting T*); do NOT use cuda::std::span's wrapped iterator.
  using iterator        = pointer;

  static constexpr bool is_device_span = is_device;

  _RAFT_HOST_DEVICE constexpr span() noexcept = default;

  _RAFT_HOST_DEVICE constexpr span(pointer ptr, size_type count) noexcept : base_{ptr, count} {}

  _RAFT_HOST_DEVICE constexpr span(pointer first, pointer last) noexcept
    : base_{first, static_cast<size_type>(last - first)}
  {
  }

  template <std::size_t N>
  _RAFT_HOST_DEVICE constexpr span(element_type (&arr)[N]) noexcept : base_{arr, N}
  {
  }

  _RAFT_HOST_DEVICE constexpr span(base_type other) noexcept : base_{other} {}

  // Qualification-converting constructor: span<U> -> span<T> when U* converts to
  // T* (e.g. span<int> -> span<const int>), matching raft::span semantics.
  template <typename U,
            std::size_t OtherExtent,
            typename = std::enable_if_t<std::is_convertible_v<U (*)[], T (*)[]> &&
                                        (Extent == dynamic_extent || Extent == OtherExtent)>>
  _RAFT_HOST_DEVICE constexpr span(span<U, is_device, OtherExtent> const& other) noexcept
    : base_{other.data(), other.size()}
  {
  }

  _RAFT_HOST_DEVICE constexpr span(span const&) noexcept            = default;
  _RAFT_HOST_DEVICE constexpr span& operator=(span const&) noexcept = default;

  _RAFT_HOST_DEVICE constexpr pointer data() const noexcept { return base_.data(); }
  _RAFT_HOST_DEVICE constexpr size_type size() const noexcept { return base_.size(); }
  _RAFT_HOST_DEVICE constexpr size_type size_bytes() const noexcept
  {
    return base_.size() * sizeof(T);
  }
  _RAFT_HOST_DEVICE constexpr bool empty() const noexcept { return base_.empty(); }

  _RAFT_HOST_DEVICE constexpr reference operator[](size_type i) const noexcept { return base_[i]; }
  _RAFT_HOST_DEVICE constexpr reference front() const noexcept { return base_.front(); }
  _RAFT_HOST_DEVICE constexpr reference back() const noexcept { return base_.back(); }

  _RAFT_HOST_DEVICE constexpr iterator begin() const noexcept { return data(); }
  _RAFT_HOST_DEVICE constexpr iterator end() const noexcept { return data() + size(); }
  _RAFT_HOST_DEVICE constexpr iterator cbegin() const noexcept { return data(); }
  _RAFT_HOST_DEVICE constexpr iterator cend() const noexcept { return data() + size(); }

  _RAFT_HOST_DEVICE constexpr span<T, is_device, dynamic_extent> subspan(
    size_type offset, size_type count = dynamic_extent) const noexcept
  {
    return span<T, is_device, dynamic_extent>{
      data() + offset, count == dynamic_extent ? size() - offset : count};
  }

 private:
  base_type base_{};
};

// Element-wise comparison, matching raft::span semantics.
template <typename T, std::size_t X, typename U, std::size_t Y, bool is_device>
_RAFT_HOST_DEVICE constexpr bool operator==(span<T, is_device, X> l, span<U, is_device, Y> r)
{
  if (l.size() != r.size()) { return false; }
  auto l_beg = l.cbegin();
  auto r_beg = r.cbegin();
  for (; l_beg != l.cend(); ++l_beg, ++r_beg) {
    if (*l_beg != *r_beg) { return false; }
  }
  return true;
}

template <typename T, std::size_t X, typename U, std::size_t Y, bool is_device>
_RAFT_HOST_DEVICE constexpr bool operator!=(span<T, is_device, X> l, span<U, is_device, Y> r)
{
  return !(l == r);
}

}  // namespace raft
