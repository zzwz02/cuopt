/*
 * cuOpt vendored RMM shim — Thrust execution policies using the pool allocator.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <rmm/cuda_stream_view.hpp>
#include <rmm/mr/per_device_resource.hpp>
#include <rmm/mr/thrust_allocator_adaptor.hpp>
#include <rmm/resource_ref.hpp>

#include <thrust/system/cuda/config.h>
#include <thrust/system/cuda/execution_policy.h>

#include <utility>

namespace rmm {

namespace detail {

using thrust_allocator_t = mr::thrust_allocator<char>;
using thrust_exec_policy_t =
  decltype(thrust::cuda::par(std::declval<thrust_allocator_t>()).on(
    std::declval<cudaStream_t>()));

using thrust_exec_policy_nosync_t =
  decltype(thrust::cuda::par_nosync(std::declval<thrust_allocator_t>()).on(
    std::declval<cudaStream_t>()));

}  // namespace detail

/** @brief Synchronous Thrust execution policy bound to a stream + pool. */
class exec_policy : public detail::thrust_exec_policy_t {
 public:
  explicit exec_policy(cuda_stream_view stream             = cuda_stream_default,
                       device_async_resource_ref mr        = mr::get_current_device_resource_ref())
    : detail::thrust_exec_policy_t(
        thrust::cuda::par(mr::thrust_allocator<char>(stream, mr)).on(stream.value()))
  {
  }
};

/** @brief Non-synchronizing Thrust execution policy bound to a stream + pool. */
class exec_policy_nosync : public detail::thrust_exec_policy_nosync_t {
 public:
  explicit exec_policy_nosync(cuda_stream_view stream      = cuda_stream_default,
                              device_async_resource_ref mr = mr::get_current_device_resource_ref())
    : detail::thrust_exec_policy_nosync_t(
        thrust::cuda::par_nosync(mr::thrust_allocator<char>(stream, mr)).on(stream.value()))
  {
  }
};

}  // namespace rmm
