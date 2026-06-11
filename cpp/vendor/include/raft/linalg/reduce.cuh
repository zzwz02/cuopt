/*
 * cuOpt vendored RAFT shim — legacy reduce (vector reduction, N==1 path).
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <raft/core/operators.hpp>
#include <raft/util/cudart_utils.hpp>

#include <rmm/device_uvector.hpp>
#include <rmm/exec_policy.hpp>

#include <cub/device/device_reduce.cuh>
#include <thrust/iterator/transform_iterator.h>

namespace raft::linalg {

namespace detail {
template <typename OutType, typename FinalLambda>
__global__ void reduce_finalize_kernel(OutType* out, const OutType* acc, FinalLambda final_op)
{
  if (threadIdx.x == 0 && blockIdx.x == 0) { out[0] = final_op(*acc); }
}
}  // namespace detail

/**
 * @brief Legacy pointer-form reduction. Supports the N==1 full-vector reduction
 * cuOpt uses: out[0] = final_op( fold(reduce_op, init, main_op(in[i])) ).
 */
template <bool rowMajor,
          bool alongRows,
          typename InType,
          typename OutType,
          typename IdxType,
          typename MainLambda   = raft::identity_op,
          typename ReduceLambda = raft::add_op,
          typename FinalLambda  = raft::identity_op>
void reduce(OutType* out,
            const InType* in,
            IdxType D,
            IdxType N,
            OutType init,
            cudaStream_t stream,
            bool /*inplace*/      = false,
            MainLambda main_op    = raft::identity_op{},
            ReduceLambda reduce_op = raft::add_op{},
            FinalLambda final_op  = raft::identity_op{})
{
  // cuOpt only uses the N==1 full-reduction form.
  IdxType const len = D * N;
  auto in_it        = thrust::make_transform_iterator(in, main_op);

  rmm::device_uvector<OutType> acc(1, stream);
  std::size_t tmp_bytes = 0;
  cub::DeviceReduce::Reduce(
    nullptr, tmp_bytes, in_it, acc.data(), static_cast<int>(len), reduce_op, init, stream);
  rmm::device_uvector<char> tmp(tmp_bytes, stream);
  cub::DeviceReduce::Reduce(
    tmp.data(), tmp_bytes, in_it, acc.data(), static_cast<int>(len), reduce_op, init, stream);

  detail::reduce_finalize_kernel<<<1, 1, 0, stream>>>(out, acc.data(), final_op);
  RAFT_CHECK_CUDA(stream);
}

}  // namespace raft::linalg
