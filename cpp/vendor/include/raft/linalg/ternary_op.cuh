/*
 * cuOpt vendored RAFT shim — legacy pointer-form ternaryOp.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <raft/util/cudart_utils.hpp>

#include <cstddef>

namespace raft::linalg {

namespace detail {

template <typename math_t, typename IdxType, typename Lambda>
__global__ void ternary_op_kernel(
  math_t* out, const math_t* in1, const math_t* in2, const math_t* in3, IdxType len, Lambda op)
{
  for (IdxType i = static_cast<IdxType>(blockIdx.x) * blockDim.x + threadIdx.x; i < len;
       i += static_cast<IdxType>(gridDim.x) * blockDim.x) {
    out[i] = op(in1[i], in2[i], in3[i]);
  }
}

}  // namespace detail

/**
 * @brief Elementwise ternary op: out[i] = op(in1[i], in2[i], in3[i]).
 * Template order matches raft (<math_t, Lambda, IdxType>) for explicit-arg call
 * sites such as ternaryOp<f_t, violation<f_t>>(...).
 */
template <typename math_t, typename Lambda, typename IdxType = int>
void ternaryOp(math_t* out,
               const math_t* in1,
               const math_t* in2,
               const math_t* in3,
               IdxType len,
               Lambda op,
               cudaStream_t stream)
{
  if (len <= 0) { return; }
  constexpr int kThreads = 256;
  const int blocks       = static_cast<int>((len + kThreads - 1) / kThreads);
  const int max_blocks   = 65535;
  detail::ternary_op_kernel<<<blocks < max_blocks ? blocks : max_blocks, kThreads, 0, stream>>>(
    out, in1, in2, in3, len, op);
  RAFT_CHECK_CUDA(stream);
}

}  // namespace raft::linalg
