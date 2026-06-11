/*
 * cuOpt vendored RAFT shim — legacy elementwise helpers.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <raft/linalg/binary_op.cuh>
#include <raft/linalg/unary_op.cuh>

#include <cstddef>

namespace raft::linalg {

namespace detail {
template <typename T>
struct sub_op {
  __host__ __device__ T operator()(T a, T b) const { return a - b; }
};
template <typename T>
struct add_op_ {
  __host__ __device__ T operator()(T a, T b) const { return a + b; }
};
template <typename T>
struct mul_op {
  __host__ __device__ T operator()(T a, T b) const { return a * b; }
};
template <typename T>
struct div_check_zero_op {
  __host__ __device__ T operator()(T a, T b) const { return b == T(0) ? T(0) : a / b; }
};
template <typename T>
struct div_scalar_op {
  T scalar;
  __host__ __device__ T operator()(T a) const { return a / scalar; }
};
}  // namespace detail

template <typename T, typename IdxType = int>
void eltwiseSub(T* out, const T* in1, const T* in2, IdxType len, cudaStream_t stream)
{
  binaryOp(out, in1, in2, len, detail::sub_op<T>{}, stream);
}

template <typename T, typename IdxType = int>
void eltwiseAdd(T* out, const T* in1, const T* in2, IdxType len, cudaStream_t stream)
{
  binaryOp(out, in1, in2, len, detail::add_op_<T>{}, stream);
}

template <typename T, typename IdxType = int>
void eltwiseMultiply(T* out, const T* in1, const T* in2, IdxType len, cudaStream_t stream)
{
  binaryOp(out, in1, in2, len, detail::mul_op<T>{}, stream);
}

template <typename T, typename IdxType = int>
void eltwiseDivideCheckZero(T* out, const T* in1, const T* in2, IdxType len, cudaStream_t stream)
{
  binaryOp(out, in1, in2, len, detail::div_check_zero_op<T>{}, stream);
}

template <typename T, typename IdxType = int>
void divideScalar(T* out, const T* in, T scalar, IdxType len, cudaStream_t stream)
{
  unaryOp(out, in, len, detail::div_scalar_op<T>{scalar}, stream);
}

}  // namespace raft::linalg
