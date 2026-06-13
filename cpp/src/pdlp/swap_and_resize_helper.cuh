/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */
#pragma once

#include <utilities/macros.cuh>

#include <rmm/device_uvector.hpp>

#include <cub/cub.cuh>

#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/permutation_iterator.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/tuple.h>
#include <thrust/universal_vector.h>

#include <cuda/std/tuple>

#include <raft/util/cuda_utils.cuh>

namespace cuopt::linear_programming::detail {

template <typename i_t>
struct swap_pair_t {
  i_t left;
  i_t right;
};

template <typename i_t>
struct matrix_swap_index_functor {
  const swap_pair_t<i_t>* pairs;
  i_t vector_size;
  bool is_left;

  HDI size_t operator()(size_t idx) const
  {
    const i_t swap_idx = static_cast<i_t>(idx / static_cast<size_t>(vector_size));
    const i_t offset   = static_cast<i_t>(idx - static_cast<size_t>(swap_idx) * vector_size);
    const i_t base     = is_left ? pairs[swap_idx].left : pairs[swap_idx].right;
    return static_cast<size_t>(base) * vector_size + offset;
  }
};

template <typename i_t, typename f_t>
__global__ void matrix_swap_kernel(f_t* matrix,
                                   const swap_pair_t<i_t>* pairs,
                                   i_t vector_size,
                                   size_t total_items)
{
  const size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx >= total_items) { return; }
  const i_t swap_idx = static_cast<i_t>(idx / static_cast<size_t>(vector_size));
  const i_t offset   = static_cast<i_t>(idx - static_cast<size_t>(swap_idx) * vector_size);
  const size_t left  = static_cast<size_t>(pairs[swap_idx].left) * vector_size + offset;
  const size_t right = static_cast<size_t>(pairs[swap_idx].right) * vector_size + offset;
  f_t tmp            = matrix[left];
  matrix[left]       = matrix[right];
  matrix[right]      = tmp;
}

template <typename i_t, typename f_t>
void matrix_swap(rmm::device_uvector<f_t>& matrix,
                 i_t vector_size,
                 const thrust::universal_host_pinned_vector<swap_pair_t<i_t>>& swap_pairs)
{
  if (swap_pairs.empty()) { return; }

  cuopt_assert(vector_size > 0, "Vector size must be greater than 0");
  cuopt_assert(matrix.size() % static_cast<size_t>(vector_size) == 0,
               "Matrix size must be divisible by vector size");
  const i_t batch_size = matrix.size() / vector_size;
  cuopt_assert(batch_size > 0, "Batch size must be greater than 0");

  const size_t swap_count  = swap_pairs.size();
  const size_t total_items = swap_count * static_cast<size_t>(vector_size);

  constexpr int block_size = 256;
  auto const grid_size =
    static_cast<unsigned int>((total_items + block_size - 1) / block_size);
  matrix_swap_kernel<i_t, f_t><<<grid_size, block_size, 0, matrix.stream().value()>>>(
    matrix.data(), thrust::raw_pointer_cast(swap_pairs.data()), vector_size, total_items);
}

template <typename host_vector_t>
void host_vector_swap(host_vector_t& host_vector, int left_swap_index, int right_swap_index)
{
  cuopt_assert(left_swap_index < host_vector.size(), "Left swap index is out of bounds");
  cuopt_assert(right_swap_index < host_vector.size(), "Right swap index is out of bounds");
  cuopt_assert(left_swap_index < right_swap_index,
               "Left swap index must be less than right swap index");

  // Swap the id to swap to the end
  std::swap(host_vector[left_swap_index], host_vector[right_swap_index]);
}
}  // namespace cuopt::linear_programming::detail
