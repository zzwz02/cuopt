/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cub/device/device_transform.cuh>

#include <cuda/std/tuple>

#include <cuda_runtime.h>

#include <type_traits>
#include <utility>

namespace cuopt {

namespace detail {

template <typename T>
struct is_cuda_std_tuple : std::false_type {
};

template <typename... Ts>
struct is_cuda_std_tuple<cuda::std::tuple<Ts...>> : std::true_type {
};

template <typename input_tuple_t,
          typename output_it_t,
          typename index_t,
          typename op_t,
          std::size_t... Is>
__device__ void tuple_transform_at(input_tuple_t const& inputs,
                                   output_it_t output,
                                   index_t i,
                                   op_t op,
                                   std::index_sequence<Is...>)
{
  *(output + i) = op((*(cuda::std::get<Is>(inputs) + i))...);
}

template <typename input_tuple_t, typename output_it_t, typename index_t, typename op_t>
__global__ void tuple_transform_kernel(input_tuple_t inputs, output_it_t output, index_t n, op_t op)
{
  auto const i = static_cast<index_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= n) { return; }

  using tuple_t = typename std::remove_cv<input_tuple_t>::type;
  tuple_transform_at(
    inputs, output, i, op, std::make_index_sequence<cuda::std::tuple_size<tuple_t>::value>{});
}

template <typename input_it_t, typename output_it_t, typename index_t, typename op_t>
__global__ void transform_kernel(input_it_t input, output_it_t output, index_t n, op_t op)
{
  auto const i = static_cast<index_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= n) { return; }
  *(output + i) = op(*(input + i));
}

}  // namespace detail

#if defined(CUOPT_USE_MACA_CCCL)

template <typename input_tuple_t, typename output_it_t, typename index_t, typename op_t>
std::enable_if_t<
  detail::is_cuda_std_tuple<typename std::remove_cv<input_tuple_t>::type>::value,
  cudaError_t>
device_transform(input_tuple_t inputs, output_it_t output, index_t n, op_t op, cudaStream_t stream)
{
  if (n <= index_t{0}) { return cudaSuccess; }

  constexpr int block_size = 256;
  auto const grid_size     = static_cast<unsigned int>(
    (static_cast<unsigned long long>(n) + block_size - 1) / block_size);
  detail::tuple_transform_kernel<<<grid_size, block_size, 0, stream>>>(inputs, output, n, op);
  return cudaGetLastError();
}

template <typename input_it_t, typename output_it_t, typename index_t, typename op_t>
std::enable_if_t<
  !detail::is_cuda_std_tuple<typename std::remove_cv<input_it_t>::type>::value,
  cudaError_t>
device_transform(input_it_t input, output_it_t output, index_t n, op_t op, cudaStream_t stream)
{
  if (n <= index_t{0}) { return cudaSuccess; }

  constexpr int block_size = 256;
  auto const grid_size     = static_cast<unsigned int>(
    (static_cast<unsigned long long>(n) + block_size - 1) / block_size);
  detail::transform_kernel<<<grid_size, block_size, 0, stream>>>(input, output, n, op);
  return cudaGetLastError();
}

#else

template <typename input_it_t, typename output_it_t, typename index_t, typename op_t>
cudaError_t device_transform(input_it_t input, output_it_t output, index_t n, op_t op, cudaStream_t stream)
{
  return cub::DeviceTransform::Transform(input, output, n, op, stream);
}

#endif

}  // namespace cuopt
