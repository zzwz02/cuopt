/*
 * cuOpt vendored RMM shim — thrust::device_vector using the pool allocator.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <rmm/mr/thrust_allocator_adaptor.hpp>

#include <thrust/device_vector.h>

namespace rmm {

/** @brief A thrust::device_vector that allocates from the current device resource. */
template <typename T>
using device_vector = thrust::device_vector<T, rmm::mr::thrust_allocator<T>>;

}  // namespace rmm
