/*
 * cuOpt vendored RMM shim — Thrust execution policies using the pool allocator.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <rmm/cuda_stream_view.hpp>
#include <rmm/mr/per_device_resource.hpp>
#include <rmm/mr/thrust_allocator_adaptor.hpp>
#include <rmm/resource_ref.hpp>

#include <thrust/system/cuda/execution_policy.h>

namespace rmm {

using thrust_exec_policy_t =
  thrust::detail::execute_with_allocator<mr::thrust_allocator<char>,
                                         thrust::cuda_cub::execute_on_stream_base>;

/** @brief Synchronous Thrust execution policy bound to a stream + pool. */
class exec_policy : public thrust_exec_policy_t {
 public:
  explicit exec_policy(cuda_stream_view stream             = cuda_stream_default,
                       device_async_resource_ref mr        = mr::get_current_device_resource_ref())
    : thrust_exec_policy_t(
        thrust::cuda::par(mr::thrust_allocator<char>(stream, mr)).on(stream.value()))
  {
  }
};

using thrust_exec_policy_nosync_t =
  thrust::detail::execute_with_allocator<mr::thrust_allocator<char>,
                                         thrust::cuda_cub::execute_on_stream_nosync_base>;

/** @brief Non-synchronizing Thrust execution policy bound to a stream + pool. */
class exec_policy_nosync : public thrust_exec_policy_nosync_t {
 public:
  explicit exec_policy_nosync(cuda_stream_view stream      = cuda_stream_default,
                              device_async_resource_ref mr = mr::get_current_device_resource_ref())
    : thrust_exec_policy_nosync_t(
        thrust::cuda::par_nosync(mr::thrust_allocator<char>(stream, mr)).on(stream.value()))
  {
  }
};

}  // namespace rmm
