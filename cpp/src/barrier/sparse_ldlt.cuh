/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */
#pragma once

#include <barrier/dense_vector.hpp>
#include <barrier/device_sparse_matrix.cuh>
#include <barrier/sparse_cholesky.cuh>

#include <dual_simplex/simplex_solver_settings.hpp>
#include <dual_simplex/sparse_matrix.hpp>
#include <dual_simplex/tic_toc.hpp>
#include <dual_simplex/types.hpp>

#include <utilities/copy_helpers.hpp>

#include <raft/core/nvtx.hpp>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <vector>

namespace cuopt::linear_programming::dual_simplex {

// Sparse LDL^T factorization of a symmetric matrix A (stored with both triangles,
// i.e. a full symmetric view) without numerical pivoting.
//
//   P * A * P^T = L * D * L^T
//
// where L is unit lower triangular (unit diagonal not stored) and D is diagonal.
// The matrices factorized by the barrier solver are either A*Dinv*A^T (normal
// equations, positive definite) or the regularized augmented KKT system
// [[-Q - D - eps_d I, A^T], [A, eps_p I]], which is symmetric quasi-definite.
// Quasi-definite matrices admit an LDL^T factorization (with mixed-sign D) for
// any symmetric permutation, so no numerical pivoting is required; ill
// conditioning is handled by the caller via adaptive regularization and
// iterative refinement.
//
// The symbolic analysis (ordering, elimination tree, symbolic factorization,
// level schedule) runs on the host. The numeric phases run either on the host
// (reference implementation) or on the device with custom kernels.
template <typename i_t, typename f_t>
class sparse_cholesky_ldlt_t : public sparse_cholesky_base_t<i_t, f_t> {
 public:
  sparse_cholesky_ldlt_t(raft::handle_t const* handle_ptr,
                         const simplex_solver_settings_t<i_t, f_t>& settings,
                         i_t size)
    : handle_ptr_(handle_ptr),
      settings_(settings),
      n_(size),
      nnz_A_(-1),
      nnz_L_(0),
      n_levels_(0),
      analyzed_(false),
      factorized_(false),
      first_factor_(true),
      positive_definite_(true)
  {
    settings_.log.printf("Sparse LDLT solver          : cuOpt built-in\n");
  }

  ~sparse_cholesky_ldlt_t() override = default;

  void set_positive_definite(bool positive_definite) override
  {
    positive_definite_ = positive_definite;
  }

  i_t analyze(device_csr_matrix_t<i_t, f_t>& Arow) override
  {
    raft::common::nvtx::range fun_scope("Barrier: LDLT Analyze");
    if (Arow.m != n_ || Arow.n != n_) {
      settings_.log.printf("Analyze input does not match size %d x %d != %d\n", Arow.m, Arow.n, n_);
      return -1;
    }
    auto stream = Arow.row_start.stream();
    i_t nnz     = Arow.row_start.element(Arow.m, stream);

    a_rowptr_host_ = cuopt::host_copy(Arow.row_start.data(), n_ + 1, stream);
    a_colidx_host_ = cuopt::host_copy(Arow.j.data(), nnz, stream);
    RAFT_CUDA_TRY(cudaStreamSynchronize(stream));

    return analyze_pattern(nnz);
  }

  i_t analyze(const csc_matrix_t<i_t, f_t>& A_in) override
  {
    raft::common::nvtx::range fun_scope("Barrier: LDLT Analyze");
    if (A_in.m != n_ || A_in.n != n_) {
      settings_.log.printf(
        "Analyze input does not match size %d x %d != %d\n", A_in.m, A_in.n, n_);
      return -1;
    }
    // A is symmetric with a full view: its CSC arrays are also valid CSR arrays.
    i_t nnz        = A_in.col_start[A_in.n];
    a_rowptr_host_ = A_in.col_start;
    a_colidx_host_.assign(A_in.i.begin(), A_in.i.begin() + nnz);

    return analyze_pattern(nnz);
  }

  i_t factorize(device_csr_matrix_t<i_t, f_t>& Arow) override
  {
    raft::common::nvtx::range fun_scope("Factorize: LDLT");
    if (!analyzed_) {
      settings_.log.printf("Factorize called before analyze\n");
      return -1;
    }
    auto stream = Arow.row_start.stream();
    i_t nnz     = Arow.row_start.element(Arow.m, stream);
    if (nnz != nnz_A_) {
      settings_.log.printf("Error: nnz %d != analyzed nnz %d\n", nnz, nnz_A_);
      return -1;
    }
    a_values_host_ = cuopt::host_copy(Arow.x.data(), nnz, stream);
    RAFT_CUDA_TRY(cudaStreamSynchronize(stream));

    return factorize_values();
  }

  i_t factorize(const csc_matrix_t<i_t, f_t>& A_in) override
  {
    raft::common::nvtx::range fun_scope("Factorize: LDLT");
    if (!analyzed_) {
      settings_.log.printf("Factorize called before analyze\n");
      return -1;
    }
    if (nnz_A_ != A_in.col_start[A_in.n]) {
      settings_.log.printf(
        "Error: nnz %d != A_in.col_start[A_in.n] %d\n", nnz_A_, A_in.col_start[A_in.n]);
      return -1;
    }
    a_values_host_.assign(A_in.x.begin(), A_in.x.begin() + nnz_A_);

    return factorize_values();
  }

  i_t solve(const dense_vector_t<i_t, f_t>& b, dense_vector_t<i_t, f_t>& x) override
  {
    if (static_cast<i_t>(b.size()) != n_ || static_cast<i_t>(x.size()) != n_) {
      settings_.log.printf("Error: solve size mismatch\n");
      return -1;
    }
    i_t status = solve_values(b.data(), x.data());
    if (status != 0) { return status; }
    for (i_t i = 0; i < n_; i++) {
      if (x[i] != x[i]) { return -1; }
    }
    return 0;
  }

  i_t solve(rmm::device_uvector<f_t>& b, rmm::device_uvector<f_t>& x) override
  {
    if (static_cast<i_t>(b.size()) != n_) {
      settings_.log.printf("Error: b.size() %d != n %d\n", static_cast<i_t>(b.size()), n_);
      return -1;
    }
    if (static_cast<i_t>(x.size()) != n_) {
      settings_.log.printf("Error: x.size() %d != n %d\n", static_cast<i_t>(x.size()), n_);
      return -1;
    }
    auto stream = b.stream();
    auto b_host = cuopt::host_copy(b.data(), n_, stream);
    std::vector<f_t> x_host(n_);
    RAFT_CUDA_TRY(cudaStreamSynchronize(stream));

    i_t status = solve_values(b_host.data(), x_host.data());
    if (status != 0) { return status; }

    raft::copy(x.data(), x_host.data(), n_, stream);
    RAFT_CUDA_TRY(cudaStreamSynchronize(stream));
    return 0;
  }

 private:
  bool halted() const
  {
    return settings_.concurrent_halt != nullptr && *settings_.concurrent_halt == 1;
  }

  // Fill-reducing ordering. perm_[k] = original index of the k-th pivot.
  // Identity ordering for now; replaced by AMD in the symbolic analysis stack.
  void compute_ordering(const std::vector<std::vector<i_t>>& upper_cols)
  {
    perm_.resize(n_);
    perm_inv_.resize(n_);
    for (i_t k = 0; k < n_; k++) {
      perm_[k]     = k;
      perm_inv_[k] = k;
    }
    (void)upper_cols;
  }

  // Symbolic analysis on the full-view symmetric pattern in
  // a_rowptr_host_/a_colidx_host_ (n_ x n_, nnz entries).
  i_t analyze_pattern(i_t nnz)
  {
    f_t start_time = tic();
    nnz_A_         = nnz;
    analyzed_      = false;
    factorized_    = false;

    if (halted()) { return CONCURRENT_HALT_RETURN; }

    // Build the strict upper-triangular pattern by column of the *unpermuted*
    // matrix once: upper_cols[c] holds rows r < c with A(r, c) != 0.
    // (The pattern is symmetric, so direction does not matter; these adjacency
    // lists are also what the ordering uses.)
    std::vector<std::vector<i_t>> upper_cols(n_);
    for (i_t r = 0; r < n_; r++) {
      for (i_t p = a_rowptr_host_[r]; p < a_rowptr_host_[r + 1]; p++) {
        i_t c = a_colidx_host_[p];
        if (r < c) {
          upper_cols[c].push_back(r);
        }
      }
    }

    compute_ordering(upper_cols);

    f_t reorder_time = toc(start_time);
    settings_.log.printf("Reordering time             : %.2fs\n", reorder_time);
    if (halted()) { return CONCURRENT_HALT_RETURN; }
    f_t start_symbolic = tic();

    // Permuted strict upper pattern by permuted column k:
    // uc[k] = sorted unique permuted rows i < k with (PAP^T)(i, k) != 0.
    std::vector<std::vector<i_t>> uc(n_);
    for (i_t c = 0; c < n_; c++) {
      for (i_t r : upper_cols[c]) {
        i_t pi = perm_inv_[r];
        i_t pk = perm_inv_[c];
        if (pi > pk) { std::swap(pi, pk); }
        uc[pk].push_back(pi);
      }
    }
    for (i_t k = 0; k < n_; k++) {
      std::sort(uc[k].begin(), uc[k].end());
      uc[k].erase(std::unique(uc[k].begin(), uc[k].end()), uc[k].end());
    }

    // Elimination tree (Liu): parent[k] = min { j > k : L(j, k) != 0 }.
    etree_parent_.assign(n_, -1);
    std::vector<i_t> ancestor(n_, -1);
    for (i_t k = 0; k < n_; k++) {
      for (i_t i : uc[k]) {
        // Walk from i up to the root of its subtree, compressing paths.
        i_t node = i;
        while (node != -1 && node != k) {
          i_t next       = ancestor[node];
          ancestor[node] = k;
          if (next == -1) { etree_parent_[node] = k; }
          node = next;
        }
      }
    }

    // Row patterns of L: rows[j] = sorted { k < j : L(j, k) != 0 }.
    // Computed via the elimination-tree reach of the entries of row j
    // (Davis, Direct Methods for Sparse Linear Systems).
    std::vector<std::vector<i_t>> rows(n_);
    {
      std::vector<i_t> mark(n_, -1);
      for (i_t j = 0; j < n_; j++) {
        mark[j] = j;
        for (i_t i : uc[j]) {
          for (i_t node = i; mark[node] != j; node = etree_parent_[node]) {
            rows[j].push_back(node);
            mark[node] = j;
          }
        }
        std::sort(rows[j].begin(), rows[j].end());
        if ((j & 1023) == 0 && halted()) { return CONCURRENT_HALT_RETURN; }
      }
    }

    // Column pointers of L (strict lower triangle, CSC).
    std::vector<i_t> col_count(n_, 0);
    int64_t nnz_l64 = 0;
    for (i_t j = 0; j < n_; j++) {
      for (i_t k : rows[j]) {
        col_count[k]++;
      }
      nnz_l64 += static_cast<int64_t>(rows[j].size());
    }
    if (nnz_l64 > static_cast<int64_t>(std::numeric_limits<i_t>::max())) {
      settings_.log.printf("Error: factor has too many nonzeros (%lld)\n",
                           static_cast<long long>(nnz_l64));
      return -1;
    }
    nnz_L_ = static_cast<i_t>(nnz_l64);
    Lp_.assign(n_ + 1, 0);
    for (i_t k = 0; k < n_; k++) {
      Lp_[k + 1] = Lp_[k] + col_count[k];
    }

    // Fill Li_ (row indices per column of L) and the row-pattern arrays.
    // Iterating j ascending keeps every column of Li_ sorted ascending, and
    // rp_pos_ records where the entry L(j, k) lives inside column k.
    Li_.resize(nnz_L_);
    rp_ptr_.assign(n_ + 1, 0);
    rp_col_.resize(nnz_L_);
    rp_pos_.resize(nnz_L_);
    {
      std::vector<i_t> fill_ptr(Lp_.begin(), Lp_.end() - 1);
      i_t t = 0;
      for (i_t j = 0; j < n_; j++) {
        rp_ptr_[j] = t;
        for (i_t k : rows[j]) {
          i_t pos      = fill_ptr[k]++;
          Li_[pos]     = j;
          rp_col_[t]   = k;
          rp_pos_[t]   = pos;
          t++;
        }
      }
      rp_ptr_[n_] = t;
    }

    // Level schedule on the elimination tree: every column of L only depends
    // on proper descendants in the etree, so columns within a level are
    // mutually independent. parent[j] > j, so one ascending pass suffices.
    {
      std::vector<i_t> level(n_, 0);
      i_t max_level = 0;
      for (i_t j = 0; j < n_; j++) {
        i_t p = etree_parent_[j];
        if (p != -1) { level[p] = std::max(level[p], level[j] + 1); }
        max_level = std::max(max_level, level[j]);
      }
      n_levels_ = max_level + 1;
      level_ptr_.assign(n_levels_ + 1, 0);
      for (i_t j = 0; j < n_; j++) {
        level_ptr_[level[j] + 1]++;
      }
      for (i_t l = 0; l < n_levels_; l++) {
        level_ptr_[l + 1] += level_ptr_[l];
      }
      level_cols_.resize(n_);
      std::vector<i_t> fill_ptr(level_ptr_.begin(), level_ptr_.end() - 1);
      for (i_t j = 0; j < n_; j++) {
        level_cols_[fill_ptr[level[j]]++] = j;
      }
    }

    if (halted()) { return CONCURRENT_HALT_RETURN; }

    // Scatter map from A entries to factor slots:
    //   a2l_[e] >= 0      : index into Lx_ (strict lower entry)
    //   a2l_[e] == -1     : entry not mapped (never happens for valid input)
    //   a2l_[e] <= -2     : diagonal entry of permuted index -(a2l_[e] + 2)
    // Both halves of a symmetric off-diagonal pair map to the same slot; the
    // numeric scatter uses plain assignment, which makes that benign.
    a2l_.assign(nnz_A_, -1);
    for (i_t r = 0; r < n_; r++) {
      for (i_t p = a_rowptr_host_[r]; p < a_rowptr_host_[r + 1]; p++) {
        i_t c  = a_colidx_host_[p];
        i_t pi = perm_inv_[r];
        i_t pk = perm_inv_[c];
        if (pi == pk) {
          a2l_[p] = -(pi + 2);
          continue;
        }
        if (pi < pk) { std::swap(pi, pk); }  // entry (row pi, col pk) of L, pi > pk
        i_t lo = Lp_[pk];
        i_t hi = Lp_[pk + 1];
        auto it = std::lower_bound(Li_.begin() + lo, Li_.begin() + hi, pi);
        if (it == Li_.begin() + hi || *it != pi) {
          settings_.log.printf("Internal error: A entry (%d, %d) missing from factor pattern\n",
                               r,
                               c);
          return -1;
        }
        a2l_[p] = static_cast<i_t>(it - Li_.begin());
      }
    }

    Lx_.assign(nnz_L_, f_t(0));
    D_.assign(n_, f_t(0));
    work_.assign(n_, f_t(0));

    f_t symbolic_time = toc(start_symbolic);
    settings_.log.printf("Symbolic factorization time : %.2fs\n", symbolic_time);
    settings_.log.printf("Symbolic nonzeros in factor : %.2e\n",
                         static_cast<f_t>(nnz_L_) + static_cast<f_t>(n_));
    settings_.log.printf("Elimination tree levels     : %d\n", n_levels_);

    analyzed_ = true;
    return 0;
  }

  // Numeric factorization (host reference implementation).
  // Left-looking by column: for each column j (ascending), apply the updates
  // of all columns k with L(j, k) != 0, then scale by the pivot d_j.
  i_t factorize_values()
  {
    f_t start_numeric = tic();
    factorized_       = false;

    // Scatter A into the factor storage (assignment, see a2l_ comment).
    std::fill(Lx_.begin(), Lx_.end(), f_t(0));
    std::fill(D_.begin(), D_.end(), f_t(0));
    for (i_t e = 0; e < nnz_A_; e++) {
      i_t m = a2l_[e];
      if (m >= 0) {
        Lx_[m] = a_values_host_[e];
      } else if (m <= -2) {
        D_[-(m + 2)] = a_values_host_[e];
      }
    }

    constexpr f_t zero_pivot_tol = 1e-300;

    for (i_t j = 0; j < n_; j++) {
      // Apply updates from all columns k with L(j, k) != 0.
      for (i_t t = rp_ptr_[j]; t < rp_ptr_[j + 1]; t++) {
        i_t k    = rp_col_[t];
        i_t p_jk = rp_pos_[t];
        f_t ljk  = Lx_[p_jk];
        f_t c    = ljk * D_[k];
        D_[j] -= c * ljk;
        // Entries of column k strictly below row j update column j.
        for (i_t p = p_jk + 1; p < Lp_[k + 1]; p++) {
          i_t i  = Li_[p];
          i_t lo = Lp_[j];
          i_t hi = Lp_[j + 1];
          auto it = std::lower_bound(Li_.begin() + lo, Li_.begin() + hi, i);
          Lx_[it - Li_.begin()] -= c * Lx_[p];
        }
      }
      // Pivot.
      f_t dj = D_[j];
      if (!std::isfinite(dj) || std::abs(dj) < zero_pivot_tol ||
          (positive_definite_ && dj <= f_t(0))) {
        settings_.log.printf("Factorization failed: pivot %e at column %d\n", dj, j);
        return -1;
      }
      for (i_t p = Lp_[j]; p < Lp_[j + 1]; p++) {
        Lx_[p] /= dj;
      }
      if ((j & 1023) == 0 && halted()) { return CONCURRENT_HALT_RETURN; }
    }

    if (first_factor_) {
      settings_.log.debug("Factorization time          : %.2fs\n", toc(start_numeric));
      first_factor_ = false;
    }
    factorized_ = true;
    return 0;
  }

  // Triangular solves (host reference implementation):
  // x = P^T L^{-T} D^{-1} L^{-1} P b.
  i_t solve_values(const f_t* b, f_t* x)
  {
    if (!factorized_) {
      settings_.log.printf("Solve called before factorize\n");
      return -1;
    }
    // work = P * b
    for (i_t k = 0; k < n_; k++) {
      work_[k] = b[perm_[k]];
    }
    // Forward solve L y = work using the rows of L (unit diagonal).
    for (i_t j = 0; j < n_; j++) {
      f_t s = work_[j];
      for (i_t t = rp_ptr_[j]; t < rp_ptr_[j + 1]; t++) {
        s -= Lx_[rp_pos_[t]] * work_[rp_col_[t]];
      }
      work_[j] = s;
    }
    // Diagonal solve.
    for (i_t j = 0; j < n_; j++) {
      work_[j] /= D_[j];
    }
    // Backward solve L^T x = work using the columns of L.
    for (i_t j = n_ - 1; j >= 0; j--) {
      f_t s = work_[j];
      for (i_t p = Lp_[j]; p < Lp_[j + 1]; p++) {
        s -= Lx_[p] * work_[Li_[p]];
      }
      work_[j] = s;
    }
    // x = P^T * work
    for (i_t k = 0; k < n_; k++) {
      x[perm_[k]] = work_[k];
    }
    return 0;
  }

  raft::handle_t const* handle_ptr_;
  const simplex_solver_settings_t<i_t, f_t>& settings_;

  i_t n_;
  i_t nnz_A_;
  i_t nnz_L_;
  i_t n_levels_;
  bool analyzed_;
  bool factorized_;
  bool first_factor_;
  bool positive_definite_;

  // Permutation: perm_[k] = original index of the k-th pivot.
  std::vector<i_t> perm_;
  std::vector<i_t> perm_inv_;

  // Elimination tree and level schedule.
  std::vector<i_t> etree_parent_;
  std::vector<i_t> level_ptr_;   // size n_levels_ + 1
  std::vector<i_t> level_cols_;  // columns grouped by level

  // Strict lower triangle of L in CSC form (row indices sorted per column).
  std::vector<i_t> Lp_;  // size n_ + 1
  std::vector<i_t> Li_;  // size nnz_L_

  // Rows of L (CSR view of the same entries):
  // row j is rp_col_[rp_ptr_[j] .. rp_ptr_[j+1]) with the position of each
  // entry inside Lx_ in rp_pos_.
  std::vector<i_t> rp_ptr_;
  std::vector<i_t> rp_col_;
  std::vector<i_t> rp_pos_;

  // Scatter map from A entries to factor slots (see analyze_pattern).
  std::vector<i_t> a2l_;

  // Numeric values.
  std::vector<f_t> Lx_;
  std::vector<f_t> D_;
  std::vector<f_t> work_;

  // Host mirror of the analyzed/factorized matrix.
  std::vector<i_t> a_rowptr_host_;
  std::vector<i_t> a_colidx_host_;
  std::vector<f_t> a_values_host_;
};

}  // namespace cuopt::linear_programming::dual_simplex
