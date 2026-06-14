/*
 * cuOpt MACA Thrust compatibility overlay.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include_next <thrust/universal_vector.h>

#if defined(CUOPT_MACA)

THRUST_NAMESPACE_BEGIN

namespace mc_cub {

template <typename T>
using universal_host_pinned_allocator =
  thrust::mr::stateless_resource_allocator<T,
                                           thrust::system::mc::universal_host_pinned_memory_resource>;

template <typename T>
using universal_host_pinned_vector =
  thrust::detail::vector_base<T, universal_host_pinned_allocator<T>>;

}  // namespace mc_cub

namespace system {
namespace mc {
using thrust::mc_cub::universal_host_pinned_allocator;
using thrust::mc_cub::universal_host_pinned_vector;
}  // namespace mc
}  // namespace system

namespace mc {
using thrust::mc_cub::universal_host_pinned_allocator;
using thrust::mc_cub::universal_host_pinned_vector;
}  // namespace mc

using thrust::system::mc::universal_host_pinned_vector;

THRUST_NAMESPACE_END

#endif
