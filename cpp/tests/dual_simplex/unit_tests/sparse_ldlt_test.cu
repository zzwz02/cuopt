/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#include <gtest/gtest.h>

#include <barrier/sparse_ldlt.cuh>

#include <dual_simplex/simplex_solver_settings.hpp>
#include <dual_simplex/sparse_matrix.hpp>

#include <utilities/copy_helpers.hpp>

#include <raft/core/handle.hpp>

#include <cmath>
#include <random>
#include <vector>

namespace cuopt::linear_programming::dual_simplex::test {

namespace {

// Builds a full-view symmetric CSC matrix from a dense column-major matrix.
csc_matrix_t<int, double> dense_to_full_csc(int n, const std::vector<double>& dense)
{
  int nnz = 0;
  for (int j = 0; j < n; j++) {
    for (int i = 0; i < n; i++) {
      if (dense[j * n + i] != 0.0) { nnz++; }
    }
  }
  csc_matrix_t<int, double> A(n, n, nnz);
  int p = 0;
  for (int j = 0; j < n; j++) {
    A.col_start[j] = p;
    for (int i = 0; i < n; i++) {
      double val = dense[j * n + i];
      if (val != 0.0) {
        A.i[p] = i;
        A.x[p] = val;
        p++;
      }
    }
  }
  A.col_start[n] = p;
  return A;
}

double residual_inf_norm(int n,
                         const std::vector<double>& dense,
                         const dense_vector_t<int, double>& x,
                         const dense_vector_t<int, double>& b)
{
  double err = 0.0;
  for (int i = 0; i < n; i++) {
    double r = b[i];
    for (int j = 0; j < n; j++) {
      r -= dense[j * n + i] * x[j];
    }
    err = std::max(err, std::abs(r));
  }
  return err;
}

// Random sparse symmetric quasi-definite KKT matrix
// [[-H, A^T], [A, eps I]] with H diagonally dominant positive definite.
std::vector<double> random_kkt_dense(int n_primal, int m_dual, unsigned seed)
{
  const int n = n_primal + m_dual;
  std::vector<double> dense(n * n, 0.0);
  std::mt19937 gen(seed);
  std::uniform_real_distribution<double> dist(-1.0, 1.0);

  // -H block (negative definite, diagonally dominant)
  for (int j = 0; j < n_primal; j++) {
    for (int i = j + 1; i < n_primal; i++) {
      if (((gen() >> 4) % 100) < 30) {
        double v          = 0.1 * dist(gen);
        dense[j * n + i]  = v;
        dense[i * n + j]  = v;
      }
    }
  }
  for (int j = 0; j < n_primal; j++) {
    double row_sum = 0.0;
    for (int i = 0; i < n_primal; i++) {
      if (i != j) { row_sum += std::abs(dense[j * n + i]); }
    }
    dense[j * n + j] = -(row_sum + 1.0 + std::abs(dist(gen)));
  }
  // A block (m_dual x n_primal), and eps I in the dual block
  for (int j = 0; j < n_primal; j++) {
    for (int i = 0; i < m_dual; i++) {
      if (((gen() >> 4) % 100) < 40) {
        double v                       = dist(gen);
        dense[j * n + (n_primal + i)]  = v;
        dense[(n_primal + i) * n + j]  = v;
      }
    }
  }
  for (int i = 0; i < m_dual; i++) {
    dense[(n_primal + i) * n + (n_primal + i)] = 1e-8;
  }
  return dense;
}

}  // namespace

TEST(sparse_ldlt, spd_host_path)
{
  raft::handle_t handle{};
  simplex_solver_settings_t<int, double> settings;

  // Small SPD matrix (2D Laplacian-like, pentadiagonal)
  const int n = 25;
  std::vector<double> dense(n * n, 0.0);
  for (int i = 0; i < n; i++) {
    dense[i * n + i] = 4.0;
    if (i + 1 < n && (i + 1) % 5 != 0) {
      dense[i * n + i + 1] = -1.0;
      dense[(i + 1) * n + i] = -1.0;
    }
    if (i + 5 < n) {
      dense[i * n + i + 5] = -1.0;
      dense[(i + 5) * n + i] = -1.0;
    }
  }
  csc_matrix_t<int, double> A = dense_to_full_csc(n, dense);

  sparse_cholesky_ldlt_t<int, double> chol(&handle, settings, n);
  chol.set_positive_definite(true);
  ASSERT_EQ(chol.analyze(A), 0);
  ASSERT_EQ(chol.factorize(A), 0);

  dense_vector_t<int, double> b(n);
  for (int i = 0; i < n; i++) {
    b[i] = 1.0 + 0.1 * i;
  }
  dense_vector_t<int, double> x(n);
  ASSERT_EQ(chol.solve(b, x), 0);
  EXPECT_LT(residual_inf_norm(n, dense, x, b), 1e-10);
}

TEST(sparse_ldlt, quasi_definite_kkt_host_path)
{
  raft::handle_t handle{};
  simplex_solver_settings_t<int, double> settings;

  const int n_primal = 40;
  const int m_dual   = 25;
  const int n        = n_primal + m_dual;
  std::vector<double> dense = random_kkt_dense(n_primal, m_dual, 1234);
  csc_matrix_t<int, double> A = dense_to_full_csc(n, dense);

  sparse_cholesky_ldlt_t<int, double> chol(&handle, settings, n);
  chol.set_positive_definite(false);  // indefinite: mixed-sign pivots expected
  ASSERT_EQ(chol.analyze(A), 0);
  ASSERT_EQ(chol.factorize(A), 0);

  dense_vector_t<int, double> b(n);
  for (int i = 0; i < n; i++) {
    b[i] = std::sin(0.7 * i) + 0.5;
  }
  dense_vector_t<int, double> x(n);
  ASSERT_EQ(chol.solve(b, x), 0);
  EXPECT_LT(residual_inf_norm(n, dense, x, b), 1e-8);
}

TEST(sparse_ldlt, device_path_and_refactorize)
{
  raft::handle_t handle{};
  auto stream = handle.get_stream();
  simplex_solver_settings_t<int, double> settings;

  const int n_primal = 30;
  const int m_dual   = 20;
  const int n        = n_primal + m_dual;
  std::vector<double> dense = random_kkt_dense(n_primal, m_dual, 99);
  csc_matrix_t<int, double> A = dense_to_full_csc(n, dense);

  // Full-view symmetric: the CSC arrays are also valid CSR arrays.
  csr_matrix_t<int, double> A_csr(n, n, A.col_start[n]);
  A.to_compressed_row(A_csr);
  device_csr_matrix_t<int, double> d_A(A_csr, stream);

  sparse_cholesky_ldlt_t<int, double> chol(&handle, settings, n);
  chol.set_positive_definite(false);
  ASSERT_EQ(chol.analyze(d_A), 0);
  ASSERT_EQ(chol.factorize(d_A), 0);

  std::vector<double> b_host(n);
  for (int i = 0; i < n; i++) {
    b_host[i] = std::cos(0.3 * i) - 0.2;
  }
  rmm::device_uvector<double> d_b(n, stream);
  rmm::device_uvector<double> d_x(n, stream);
  raft::copy(d_b.data(), b_host.data(), n, stream);
  ASSERT_EQ(chol.solve(d_b, d_x), 0);

  auto x_host = cuopt::host_copy(d_x.data(), n, stream);
  RAFT_CUDA_TRY(cudaStreamSynchronize(stream));
  dense_vector_t<int, double> x_vec(std::vector<double>(x_host.begin(), x_host.end()));
  dense_vector_t<int, double> b_vec(b_host);
  EXPECT_LT(residual_inf_norm(n, dense, x_vec, b_vec), 1e-8);

  // Refactorize with modified values on the same pattern.
  std::vector<double> dense2 = dense;
  for (int j = 0; j < n_primal; j++) {
    dense2[j * n + j] *= 2.0;
  }
  csc_matrix_t<int, double> A2 = dense_to_full_csc(n, dense2);
  csr_matrix_t<int, double> A2_csr(n, n, A2.col_start[n]);
  A2.to_compressed_row(A2_csr);
  raft::copy(d_A.x.data(), A2_csr.x.data(), A2_csr.x.size(), stream);
  RAFT_CUDA_TRY(cudaStreamSynchronize(stream));

  ASSERT_EQ(chol.factorize(d_A), 0);
  ASSERT_EQ(chol.solve(d_b, d_x), 0);
  auto x2_host = cuopt::host_copy(d_x.data(), n, stream);
  RAFT_CUDA_TRY(cudaStreamSynchronize(stream));
  dense_vector_t<int, double> x2_vec(std::vector<double>(x2_host.begin(), x2_host.end()));
  EXPECT_LT(residual_inf_norm(n, dense2, x2_vec, b_vec), 1e-8);
}

TEST(sparse_ldlt, singular_matrix_fails)
{
  raft::handle_t handle{};
  simplex_solver_settings_t<int, double> settings;

  const int n = 3;
  // Rank-deficient matrix: third row/col is zero except a structural diagonal 0.
  std::vector<double> dense = {2.0, 1.0, 0.0, 1.0, 2.0, 0.0, 0.0, 0.0, 0.0};
  // Add explicit structural zero on the diagonal so the pattern is valid.
  csc_matrix_t<int, double> A(n, n, 5);
  A.col_start = {0, 2, 4, 5};
  A.i         = {0, 1, 0, 1, 2};
  A.x         = {2.0, 1.0, 1.0, 2.0, 0.0};

  sparse_cholesky_ldlt_t<int, double> chol(&handle, settings, n);
  chol.set_positive_definite(false);
  ASSERT_EQ(chol.analyze(A), 0);
  EXPECT_EQ(chol.factorize(A), -1);
}

}  // namespace cuopt::linear_programming::dual_simplex::test
