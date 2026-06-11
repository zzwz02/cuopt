/*
 * cuOpt vendored RAFT shim — legacy pointer-form binaryOp.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <raft/util/cudart_utils.hpp>

#include <cstddef>
#include <cstdint>

namespace raft::linalg {

namespace detail {

template <typename OutT, typename InT, typename IdxType, typename Lambda>
__global__ void binary_op_kernel(OutT* out, const InT* in1, const InT* in2, IdxType len, Lambda op)
{
  for (IdxType i = static_cast<IdxType>(blockIdx.x) * blockDim.x + threadIdx.x; i < len;
       i += static_cast<IdxType>(gridDim.x) * blockDim.x) {
    out[i] = op(in1[i], in2[i]);
  }
}

}  // namespace detail

/**
 * @brief Elementwise binary op: out[i] = op(in1[i], in2[i]).
 * Legacy pointer-form signature used by cuOpt.
 */
template <typename OutT, typename InT, typename IdxType, typename Lambda>
void binaryOp(OutT* out, const InT* in1, const InT* in2, IdxType len, Lambda op, cudaStream_t stream)
{
  if (len <= 0) { return; }
  constexpr int kThreads = 256;
  const int blocks       = static_cast<int>((len + kThreads - 1) / kThreads);
  const int max_blocks   = 65535;
  detail::binary_op_kernel<<<blocks < max_blocks ? blocks : max_blocks, kThreads, 0, stream>>>(
    out, in1, in2, len, op);
  RAFT_CHECK_CUDA(stream);
}

}  // namespace raft::linalg
