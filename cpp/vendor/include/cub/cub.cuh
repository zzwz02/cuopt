/*
 * cuOpt MACA CUB compatibility overlay.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include_next <cub/cub.cuh>

#include <utilities/device_transform.cuh>

#if defined(CUOPT_USE_MACA_CCCL)

namespace cub {
namespace detail {

template <typename offset_t, typename function_t>
__global__ void device_for_bulk_kernel(offset_t n, function_t f)
{
  auto const idx = static_cast<offset_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx < n) { f(idx); }
}

}  // namespace detail

struct DeviceFor {
  template <typename offset_t, typename function_t>
  static cudaError_t Bulk(offset_t n, function_t f, cudaStream_t stream = 0)
  {
    if (n <= 0) { return cudaSuccess; }
    constexpr int block_size = 256;
    auto const grid_size =
      static_cast<unsigned int>((static_cast<unsigned long long>(n) + block_size - 1) / block_size);
    detail::device_for_bulk_kernel<<<grid_size, block_size, 0, stream>>>(n, f);
    return cudaGetLastError();
  }
};

}  // namespace cub

#endif
