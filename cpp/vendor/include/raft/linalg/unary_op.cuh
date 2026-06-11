/*
 * cuOpt vendored RAFT shim — legacy pointer-form unaryOp.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <raft/util/cudart_utils.hpp>

#include <cstddef>

namespace raft::linalg {

namespace detail {

template <typename OutT, typename InT, typename IdxType, typename Lambda>
__global__ void unary_op_kernel(OutT* out, const InT* in, IdxType len, Lambda op)
{
  for (IdxType i = static_cast<IdxType>(blockIdx.x) * blockDim.x + threadIdx.x; i < len;
       i += static_cast<IdxType>(gridDim.x) * blockDim.x) {
    out[i] = op(in[i]);
  }
}

}  // namespace detail

/** @brief Elementwise unary op: out[i] = op(in[i]). */
template <typename OutT, typename InT, typename IdxType, typename Lambda>
void unaryOp(OutT* out, const InT* in, IdxType len, Lambda op, cudaStream_t stream)
{
  if (len <= 0) { return; }
  constexpr int kThreads = 256;
  const int blocks       = static_cast<int>((len + kThreads - 1) / kThreads);
  const int max_blocks   = 65535;
  detail::unary_op_kernel<<<blocks < max_blocks ? blocks : max_blocks, kThreads, 0, stream>>>(
    out, in, len, op);
  RAFT_CHECK_CUDA(stream);
}

}  // namespace raft::linalg
