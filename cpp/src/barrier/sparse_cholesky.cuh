/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */
#pragma once

#include <barrier/dense_vector.hpp>
#include <barrier/device_sparse_matrix.cuh>

#include <dual_simplex/sparse_matrix.hpp>

#include <rmm/device_uvector.hpp>

namespace cuopt::linear_programming::dual_simplex {

// Interface for the sparse symmetric factorization used by the barrier solver
// to solve the normal-equations (A*Dinv*A^T) or augmented KKT systems.
template <typename i_t, typename f_t>
class sparse_cholesky_base_t {
 public:
  virtual ~sparse_cholesky_base_t()                                                 = default;
  virtual i_t analyze(const csc_matrix_t<i_t, f_t>& A_in)                           = 0;
  virtual i_t factorize(const csc_matrix_t<i_t, f_t>& A_in)                         = 0;
  virtual i_t analyze(device_csr_matrix_t<i_t, f_t>& A_in)                          = 0;
  virtual i_t factorize(device_csr_matrix_t<i_t, f_t>& A_in)                        = 0;
  virtual i_t solve(const dense_vector_t<i_t, f_t>& b, dense_vector_t<i_t, f_t>& x) = 0;
  virtual i_t solve(rmm::device_uvector<f_t>& b, rmm::device_uvector<f_t>& x)       = 0;
  virtual void set_positive_definite(bool positive_definite)                        = 0;

  // Optional structured pivot handling. n_negative == 0: the matrix is SPD
  // (normal equations) and numerically rank-deficient pivots may be dropped.
  // n_negative > 0: the matrix is symmetric quasi-definite; original indices
  // < n_negative expect negative pivots (floored at negative_floor), the rest
  // positive ones (floored at positive_floor). Default: unsigned static
  // pivoting.
  virtual void set_pivot_structure(i_t n_negative, f_t negative_floor, f_t positive_floor) {}

  // Expected-sign violations corrected during the last factorize (0 when the
  // implementation does not track them).
  virtual i_t pivot_corrections() const { return 0; }
};

}  // namespace cuopt::linear_programming::dual_simplex
