/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved. SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once

#include <rmm/device_uvector.hpp>

#include <cub/cub.cuh>
#include <cuda_runtime.h>

namespace cuopt::linear_programming::detail {

template <typename f_t>
struct fixed_size_segmented_sum_op {
  __host__ __device__ f_t operator()(f_t a, f_t b) const { return a + b; }
};

template <typename i_t, typename f_t, typename InputIteratorT, typename OutputIteratorT, typename ReductionOpT>
__global__ void fixed_size_segmented_reduce_kernel(InputIteratorT input,
                                                   OutputIteratorT output,
                                                   i_t batch_size,
                                                   i_t problem_size,
                                                   ReductionOpT reduction_op,
                                                   f_t initial_value)
{
  constexpr int block_size = 256;
  __shared__ f_t shared[block_size];

  auto const segment = static_cast<i_t>(blockIdx.x);
  if (segment >= batch_size) { return; }

  f_t thread_value = initial_value;
  auto const base  = static_cast<size_t>(segment) * static_cast<size_t>(problem_size);
  for (i_t i = static_cast<i_t>(threadIdx.x); i < problem_size; i += block_size) {
    thread_value = reduction_op(thread_value, *(input + base + static_cast<size_t>(i)));
  }

  shared[threadIdx.x] = thread_value;
  __syncthreads();

  for (int offset = block_size / 2; offset > 0; offset >>= 1) {
    if (threadIdx.x < offset) {
      shared[threadIdx.x] = reduction_op(shared[threadIdx.x], shared[threadIdx.x + offset]);
    }
    __syncthreads();
  }

  if (threadIdx.x == 0) { *(output + segment) = shared[0]; }
}

template <typename i_t, typename f_t, typename InputIteratorT, typename OutputIteratorT, typename ReductionOpT>
cudaError_t fixed_size_segmented_reduce(InputIteratorT input,
                                        OutputIteratorT output,
                                        i_t batch_size,
                                        i_t problem_size,
                                        ReductionOpT reduction_op,
                                        f_t initial_value,
                                        rmm::cuda_stream_view stream_view)
{
  if (batch_size <= i_t{0}) { return cudaSuccess; }

  constexpr int block_size = 256;
  fixed_size_segmented_reduce_kernel<i_t, f_t><<<static_cast<unsigned int>(batch_size),
                                                 block_size,
                                                 0,
                                                 stream_view.value()>>>(
    input, output, batch_size, problem_size, reduction_op, initial_value);
  return cudaGetLastError();
}

template <typename i_t, typename f_t, typename InputIteratorT, typename OutputIteratorT>
cudaError_t fixed_size_segmented_sum(InputIteratorT input,
                                     OutputIteratorT output,
                                     i_t batch_size,
                                     i_t problem_size,
                                     rmm::cuda_stream_view stream_view)
{
  return fixed_size_segmented_reduce<i_t, f_t>(
    input, output, batch_size, problem_size, fixed_size_segmented_sum_op<f_t>{}, f_t{0}, stream_view);
}

template <typename i_t, typename f_t>
struct segmented_sum_handler_t {
  segmented_sum_handler_t(rmm::cuda_stream_view stream_view) : stream_view_(stream_view) {}

  template <typename InputIteratorT, typename OutputIteratorT>
  void segmented_sum_helper(InputIteratorT input,
                            OutputIteratorT output,
                            i_t batch_size,
                            i_t problem_size)
  {
#if defined(CUOPT_USE_MACA_CCCL)
    byte_needed_ = 0;
    segmented_sum_storage_.resize(0, stream_view_);
    fixed_size_segmented_sum<i_t, f_t>(input, output, batch_size, problem_size, stream_view_);
#else
    cub::DeviceSegmentedReduce::Sum(
      nullptr, byte_needed_, input, output, batch_size, problem_size, stream_view_);

    segmented_sum_storage_.resize(byte_needed_, stream_view_);

    cub::DeviceSegmentedReduce::Sum(segmented_sum_storage_.data(),
                                    byte_needed_,
                                    input,
                                    output,
                                    batch_size,
                                    problem_size,
                                    stream_view_);
#endif
  }

  template <typename InputIteratorT, typename ReductionOpT>
  void segmented_reduce_helper(InputIteratorT input,
                               f_t* output,
                               i_t batch_size,
                               i_t problem_size,
                               ReductionOpT reduction_op,
                               f_t initial_value)
  {
#if defined(CUOPT_USE_MACA_CCCL)
    byte_needed_ = 0;
    segmented_sum_storage_.resize(0, stream_view_.value());
    fixed_size_segmented_reduce<i_t, f_t>(
      input, output, batch_size, problem_size, reduction_op, initial_value, stream_view_);
#else
    cub::DeviceSegmentedReduce::Reduce(nullptr,
                                       byte_needed_,
                                       input,
                                       output,
                                       batch_size,
                                       problem_size,
                                       reduction_op,
                                       initial_value,
                                       stream_view_.value());

    segmented_sum_storage_.resize(byte_needed_, stream_view_.value());

    cub::DeviceSegmentedReduce::Reduce(segmented_sum_storage_.data(),
                                       byte_needed_,
                                       input,
                                       output,
                                       batch_size,
                                       problem_size,
                                       reduction_op,
                                       initial_value,
                                       stream_view_.value());
#endif
  }

  size_t byte_needed_;
  rmm::device_buffer segmented_sum_storage_;
  rmm::cuda_stream_view stream_view_;
};

}  // namespace cuopt::linear_programming::detail
