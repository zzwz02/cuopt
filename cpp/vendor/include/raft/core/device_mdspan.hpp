/*
 * cuOpt vendored RAFT shim — device_mdspan over cuda::std::mdspan.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <cuda/std/mdspan>

#include <cstddef>

namespace raft {

using cuda::std::dynamic_extent;
using cuda::std::layout_left;
using cuda::std::layout_right;

template <typename IndexType, std::size_t... Extents>
using extents = cuda::std::extents<IndexType, Extents...>;

using row_major = cuda::std::layout_right;
using col_major = cuda::std::layout_left;

template <typename ElementType,
          typename Extents,
          typename LayoutPolicy   = cuda::std::layout_right,
          typename AccessorPolicy = cuda::std::default_accessor<ElementType>>
using device_mdspan = cuda::std::mdspan<ElementType, Extents, LayoutPolicy, AccessorPolicy>;

template <typename ElementType,
          typename Extents,
          typename LayoutPolicy   = cuda::std::layout_right,
          typename AccessorPolicy = cuda::std::default_accessor<ElementType>>
using host_mdspan = cuda::std::mdspan<ElementType, Extents, LayoutPolicy, AccessorPolicy>;

/** @brief Build a device_mdspan from a raw pointer and an extents object. */
template <typename ElementType, typename IndexType, std::size_t... Extents>
constexpr auto make_mdspan(ElementType* ptr, extents<IndexType, Extents...> exts)
{
  return device_mdspan<ElementType, extents<IndexType, Extents...>>(ptr, exts);
}

}  // namespace raft
