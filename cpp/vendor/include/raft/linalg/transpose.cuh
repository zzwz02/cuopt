/*
 * cuOpt vendored RAFT shim — dense transpose via cuBLAS geam (pointer form).
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <raft/core/cublas_macros.hpp>
#include <raft/core/handle.hpp>

#include <cublas_v2.h>

namespace raft::linalg {

namespace detail {
inline cublasStatus_t cublasgeam_dispatch(cublasHandle_t h,
                                          cublasOperation_t transa,
                                          cublasOperation_t transb,
                                          int m,
                                          int n,
                                          const float* alpha,
                                          const float* A,
                                          int lda,
                                          const float* beta,
                                          const float* B,
                                          int ldb,
                                          float* C,
                                          int ldc)
{
  return cublasSgeam(h, transa, transb, m, n, alpha, A, lda, beta, B, ldb, C, ldc);
}
inline cublasStatus_t cublasgeam_dispatch(cublasHandle_t h,
                                          cublasOperation_t transa,
                                          cublasOperation_t transb,
                                          int m,
                                          int n,
                                          const double* alpha,
                                          const double* A,
                                          int lda,
                                          const double* beta,
                                          const double* B,
                                          int ldb,
                                          double* C,
                                          int ldc)
{
  return cublasDgeam(h, transa, transb, m, n, alpha, A, lda, beta, B, ldb, C, ldc);
}
}  // namespace detail

/**
 * @brief Transpose an n_rows x n_cols matrix: out = in^T.
 * Mirrors raft's pointer-form transpose (cuBLAS geam, column-major).
 */
template <typename math_t>
void transpose(raft::resources const& handle,
               math_t* in,
               math_t* out,
               int n_rows,
               int n_cols,
               cudaStream_t stream)
{
  int const out_n_rows = n_cols;
  int const out_n_cols = n_rows;
  cublasHandle_t cublas_h = raft::resource::get_cublas_handle(handle);
  RAFT_CUBLAS_TRY(cublasSetStream(cublas_h, stream));
  math_t const alpha = 1;
  math_t const beta  = 0;
  RAFT_CUBLAS_TRY(detail::cublasgeam_dispatch(cublas_h,
                                              CUBLAS_OP_T,
                                              CUBLAS_OP_N,
                                              out_n_rows,
                                              out_n_cols,
                                              &alpha,
                                              in,
                                              n_rows,
                                              &beta,
                                              out,
                                              out_n_rows,
                                              out,
                                              out_n_rows));
}

}  // namespace raft::linalg
