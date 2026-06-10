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

#include <rmm/device_scalar.hpp>
#include <rmm/device_uvector.hpp>

#include <amd.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <vector>

namespace cuopt::linear_programming::dual_simplex {

namespace ldlt_detail {

constexpr int factor_block_dim = 128;

// Scatters the values of the analyzed matrix into the factor storage.
// Lx and D must be zeroed beforehand: slots that receive no A entry are
// structural fill. Both halves of a symmetric off-diagonal pair map to the
// same slot and write the same value, so the race is benign.
template <typename i_t, typename f_t>
__global__ void ldlt_scatter_kernel(
  const f_t* a_values, const i_t* a2l, i_t nnz_A, f_t* Lx, f_t* D)
{
  const i_t e = static_cast<i_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (e >= nnz_A) { return; }
  const i_t m = a2l[e];
  if (m >= 0) {
    Lx[m] = a_values[e];
  } else if (m <= -2) {
    D[-(m + 2)] = a_values[e];
  }
}

// Left-looking factorization of one level: every block owns one column j of
// the level and applies the updates of all columns k with L(j, k) != 0, which
// are complete because they live in lower levels. The k loop is sequential
// within the block (deterministic accumulation order, identical to the host
// reference); threads parallelize over the entries of column k and scatter
// into column j via binary search in its sorted row indices.
template <typename i_t, typename f_t>
__global__ void ldlt_factor_level_kernel(const i_t* level_cols,
                                         i_t level_start,
                                         const i_t* Lp,
                                         const i_t* Li,
                                         const i_t* rp_ptr,
                                         const i_t* rp_col,
                                         const i_t* rp_pos,
                                         f_t* Lx,
                                         f_t* D,
                                         i_t* fail_flag,
                                         bool positive_definite)
{
  const i_t j       = level_cols[level_start + static_cast<i_t>(blockIdx.x)];
  const i_t col_beg = Lp[j];
  const i_t col_end = Lp[j + 1];

  for (i_t t = rp_ptr[j]; t < rp_ptr[j + 1]; t++) {
    const i_t k    = rp_col[t];
    const i_t p_jk = rp_pos[t];
    const f_t ljk  = Lx[p_jk];
    const f_t c    = ljk * D[k];
    if (threadIdx.x == 0) { D[j] -= c * ljk; }
    const i_t k_end = Lp[k + 1];
    for (i_t p = p_jk + 1 + static_cast<i_t>(threadIdx.x); p < k_end;
         p += static_cast<i_t>(blockDim.x)) {
      const i_t i = Li[p];
      i_t lo      = col_beg;
      i_t hi      = col_end;
      while (lo < hi) {
        const i_t mid = lo + (hi - lo) / 2;
        if (Li[mid] < i) {
          lo = mid + 1;
        } else {
          hi = mid;
        }
      }
      Lx[lo] -= c * Lx[p];
    }
    // The next k iteration scatters into the same column; updates of distinct
    // k may target the same entry, so the block must advance in lockstep.
    __syncthreads();
  }

  const f_t dj = D[j];
  if (!isfinite(dj) || fabs(dj) < f_t(1e-300) || (positive_definite && dj <= f_t(0))) {
    if (threadIdx.x == 0) { atomicCAS(fail_flag, i_t(0), j + 1); }
    return;
  }
  for (i_t p = col_beg + static_cast<i_t>(threadIdx.x); p < col_end;
       p += static_cast<i_t>(blockDim.x)) {
    Lx[p] /= dj;
  }
}

// y[k] = b[perm[k]]
template <typename i_t, typename f_t>
__global__ void ldlt_perm_gather_kernel(const f_t* b, const i_t* perm, i_t n, f_t* y)
{
  const i_t k = static_cast<i_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (k < n) { y[k] = b[perm[k]]; }
}

// x[perm[k]] = y[k]
template <typename i_t, typename f_t>
__global__ void ldlt_perm_scatter_kernel(const f_t* y, const i_t* perm, i_t n, f_t* x)
{
  const i_t k = static_cast<i_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (k < n) { x[perm[k]] = y[k]; }
}

// y[j] /= D[j]
template <typename i_t, typename f_t>
__global__ void ldlt_diag_scale_kernel(const f_t* D, i_t n, f_t* y)
{
  const i_t j = static_cast<i_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (j < n) { y[j] /= D[j]; }
}

// Fixed-order tree reduction over the block: deterministic for a fixed block
// size. sdata must hold blockDim.x elements.
template <typename f_t>
__device__ inline f_t ldlt_block_reduce(f_t partial, f_t* sdata)
{
  sdata[threadIdx.x] = partial;
  __syncthreads();
  for (unsigned s = blockDim.x / 2; s > 0; s >>= 1) {
    if (threadIdx.x < s) { sdata[threadIdx.x] += sdata[threadIdx.x + s]; }
    __syncthreads();
  }
  return sdata[0];
}

// Forward substitution L y = b for one level, row form (unit diagonal):
// y_j -= sum_{k < j, L(j,k) != 0} L(j,k) * y_k. Every k in the row pattern is
// a proper etree descendant of j, hence in a lower level and already final.
template <typename i_t, typename f_t>
__global__ void ldlt_forward_level_kernel(const i_t* level_cols,
                                          i_t level_start,
                                          const i_t* rp_ptr,
                                          const i_t* rp_col,
                                          const i_t* rp_pos,
                                          const f_t* Lx,
                                          f_t* y)
{
  __shared__ f_t sdata[factor_block_dim];
  const i_t j   = level_cols[level_start + static_cast<i_t>(blockIdx.x)];
  const i_t beg = rp_ptr[j];
  const i_t end = rp_ptr[j + 1];
  f_t partial   = f_t(0);
  for (i_t t = beg + static_cast<i_t>(threadIdx.x); t < end; t += static_cast<i_t>(blockDim.x)) {
    partial += Lx[rp_pos[t]] * y[rp_col[t]];
  }
  const f_t s = ldlt_block_reduce(partial, sdata);
  if (threadIdx.x == 0) { y[j] -= s; }
}

// Backward substitution L^T x = y for one level, column form:
// x_j -= sum_{i > j, L(i,j) != 0} L(i,j) * x_i. Every i in the column pattern
// is a proper etree ancestor of j, hence in a higher level; levels run in
// reverse order.
template <typename i_t, typename f_t>
__global__ void ldlt_backward_level_kernel(const i_t* level_cols,
                                           i_t level_start,
                                           const i_t* Lp,
                                           const i_t* Li,
                                           const f_t* Lx,
                                           f_t* y)
{
  __shared__ f_t sdata[factor_block_dim];
  const i_t j   = level_cols[level_start + static_cast<i_t>(blockIdx.x)];
  const i_t beg = Lp[j];
  const i_t end = Lp[j + 1];
  f_t partial   = f_t(0);
  for (i_t p = beg + static_cast<i_t>(threadIdx.x); p < end; p += static_cast<i_t>(blockDim.x)) {
    partial += Lx[p] * y[Li[p]];
  }
  const f_t s = ldlt_block_reduce(partial, sdata);
  if (threadIdx.x == 0) { y[j] -= s; }
}

}  // namespace ldlt_detail

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
// The symbolic analysis (AMD ordering, elimination tree, symbolic
// factorization, level schedule) runs on the host. The numeric factorization
// runs on the device with custom level-scheduled kernels (deterministic by
// construction); setting CUOPT_LDLT_HOST=1 in the environment switches to the
// host reference implementation of the numeric phases.
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
      positive_definite_(true),
      use_host_numeric_(std::getenv("CUOPT_LDLT_HOST") != nullptr),
      d_Lp_(0, handle_ptr->get_stream()),
      d_Li_(0, handle_ptr->get_stream()),
      d_rp_ptr_(0, handle_ptr->get_stream()),
      d_rp_col_(0, handle_ptr->get_stream()),
      d_rp_pos_(0, handle_ptr->get_stream()),
      d_a2l_(0, handle_ptr->get_stream()),
      d_level_cols_(0, handle_ptr->get_stream()),
      d_perm_(0, handle_ptr->get_stream()),
      d_Lx_(0, handle_ptr->get_stream()),
      d_D_(0, handle_ptr->get_stream()),
      d_a_values_(0, handle_ptr->get_stream()),
      d_work_(0, handle_ptr->get_stream()),
      d_fail_(handle_ptr->get_stream())
  {
    settings_.log.printf("Sparse LDLT solver          : cuOpt built-in (%s numeric)\n",
                         use_host_numeric_ ? "host" : "device");
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

    if (use_host_numeric_) {
      a_values_host_ = cuopt::host_copy(Arow.x.data(), nnz, stream);
      RAFT_CUDA_TRY(cudaStreamSynchronize(stream));
      return factorize_values_host();
    }
    return factorize_values_device(Arow.x.data());
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

    if (use_host_numeric_) {
      a_values_host_.assign(A_in.x.begin(), A_in.x.begin() + nnz_A_);
      return factorize_values_host();
    }
    auto stream = handle_ptr_->get_stream();
    d_a_values_.resize(nnz_A_, stream);
    raft::copy(d_a_values_.data(), A_in.x.data(), nnz_A_, stream);
    return factorize_values_device(d_a_values_.data());
  }

  i_t solve(const dense_vector_t<i_t, f_t>& b, dense_vector_t<i_t, f_t>& x) override
  {
    if (static_cast<i_t>(b.size()) != n_ || static_cast<i_t>(x.size()) != n_) {
      settings_.log.printf("Error: solve size mismatch\n");
      return -1;
    }
    i_t status;
    if (use_host_numeric_) {
      status = solve_values_host(b.data(), x.data());
    } else {
      auto stream = handle_ptr_->get_stream();
      rmm::device_uvector<f_t> d_b(n_, stream);
      rmm::device_uvector<f_t> d_x(n_, stream);
      raft::copy(d_b.data(), b.data(), n_, stream);
      status = solve_values_device(d_b.data(), d_x.data(), stream);
      if (status == 0) {
        raft::copy(x.data(), d_x.data(), n_, stream);
        RAFT_CUDA_TRY(cudaStreamSynchronize(stream));
      }
    }
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
    if (!use_host_numeric_) { return solve_values_device(b.data(), x.data(), stream); }

    auto b_host = cuopt::host_copy(b.data(), n_, stream);
    std::vector<f_t> x_host(n_);
    RAFT_CUDA_TRY(cudaStreamSynchronize(stream));

    i_t status = solve_values_host(b_host.data(), x_host.data());
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
  // Uses SuiteSparse AMD (approximate minimum degree) on the full symmetric
  // pattern; the ordering only affects fill-in, never correctness, so any
  // failure falls back to the natural ordering.
  void compute_ordering()
  {
    perm_.resize(n_);
    perm_inv_.resize(n_);
    for (i_t k = 0; k < n_; k++) {
      perm_[k] = k;
    }

    static_assert(std::is_same_v<i_t, int>, "amd_order requires int32 indices");
    if (n_ > 1) {
      i_t status = amd_order(
        n_, a_rowptr_host_.data(), a_colidx_host_.data(), perm_.data(), nullptr, nullptr);
      if (status != AMD_OK && status != AMD_OK_BUT_JUMBLED) {
        settings_.log.printf("AMD ordering failed (%d); using natural ordering\n", status);
        for (i_t k = 0; k < n_; k++) {
          perm_[k] = k;
        }
      }
    }

    for (i_t k = 0; k < n_; k++) {
      perm_inv_[perm_[k]] = k;
    }
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

    compute_ordering();

    f_t reorder_time = toc(start_time);
    settings_.log.printf("Reordering time             : %.2fs\n", reorder_time);
    if (halted()) { return CONCURRENT_HALT_RETURN; }
    f_t start_symbolic = tic();

    // Permuted strict upper pattern by permuted column k:
    // uc[k] = sorted unique permuted rows i < k with (PAP^T)(i, k) != 0.
    // The input pattern is symmetric, so visiting each strict upper entry
    // (r < c) once covers every off-diagonal pair.
    std::vector<std::vector<i_t>> uc(n_);
    for (i_t r = 0; r < n_; r++) {
      for (i_t p = a_rowptr_host_[r]; p < a_rowptr_host_[r + 1]; p++) {
        i_t c = a_colidx_host_[p];
        if (r >= c) { continue; }
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
          i_t pos    = fill_ptr[k]++;
          Li_[pos]   = j;
          rp_col_[t] = k;
          rp_pos_[t] = pos;
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
        i_t lo  = Lp_[pk];
        i_t hi  = Lp_[pk + 1];
        auto it = std::lower_bound(Li_.begin() + lo, Li_.begin() + hi, pi);
        if (it == Li_.begin() + hi || *it != pi) {
          settings_.log.printf(
            "Internal error: A entry (%d, %d) missing from factor pattern\n", r, c);
          return -1;
        }
        a2l_[p] = static_cast<i_t>(it - Li_.begin());
      }
    }

    Lx_.assign(nnz_L_, f_t(0));
    D_.assign(n_, f_t(0));
    work_.assign(n_, f_t(0));

    // Upload the symbolic structures for the device numeric phases.
    if (!use_host_numeric_) {
      auto stream = handle_ptr_->get_stream();
      d_Lp_.resize(n_ + 1, stream);
      d_Li_.resize(nnz_L_, stream);
      d_rp_ptr_.resize(n_ + 1, stream);
      d_rp_col_.resize(nnz_L_, stream);
      d_rp_pos_.resize(nnz_L_, stream);
      d_a2l_.resize(nnz_A_, stream);
      d_level_cols_.resize(n_, stream);
      d_perm_.resize(n_, stream);
      d_Lx_.resize(nnz_L_, stream);
      d_D_.resize(n_, stream);
      d_work_.resize(n_, stream);
      raft::copy(d_Lp_.data(), Lp_.data(), Lp_.size(), stream);
      raft::copy(d_Li_.data(), Li_.data(), Li_.size(), stream);
      raft::copy(d_rp_ptr_.data(), rp_ptr_.data(), rp_ptr_.size(), stream);
      raft::copy(d_rp_col_.data(), rp_col_.data(), rp_col_.size(), stream);
      raft::copy(d_rp_pos_.data(), rp_pos_.data(), rp_pos_.size(), stream);
      raft::copy(d_a2l_.data(), a2l_.data(), a2l_.size(), stream);
      raft::copy(d_level_cols_.data(), level_cols_.data(), level_cols_.size(), stream);
      raft::copy(d_perm_.data(), perm_.data(), perm_.size(), stream);
      RAFT_CUDA_TRY(cudaStreamSynchronize(stream));
    }

    f_t symbolic_time = toc(start_symbolic);
    settings_.log.printf("Symbolic factorization time : %.2fs\n", symbolic_time);
    settings_.log.printf("Symbolic nonzeros in factor : %.2e\n",
                         static_cast<f_t>(nnz_L_) + static_cast<f_t>(n_));
    settings_.log.printf("Elimination tree levels     : %d\n", n_levels_);

    analyzed_ = true;
    return 0;
  }

  // Numeric factorization on the device with level-scheduled kernels.
  // a_values is a device pointer holding the nnz_A_ values of the analyzed
  // matrix in its original entry order.
  i_t factorize_values_device(const f_t* a_values)
  {
    f_t start_numeric = tic();
    factorized_       = false;
    auto stream       = handle_ptr_->get_stream();

    RAFT_CUDA_TRY(cudaMemsetAsync(d_Lx_.data(), 0, sizeof(f_t) * nnz_L_, stream));
    RAFT_CUDA_TRY(cudaMemsetAsync(d_D_.data(), 0, sizeof(f_t) * n_, stream));
    d_fail_.set_value_to_zero_async(stream);

    constexpr int block_dim = ldlt_detail::factor_block_dim;
    if (nnz_A_ > 0) {
      const int grid = (nnz_A_ + block_dim - 1) / block_dim;
      ldlt_detail::ldlt_scatter_kernel<i_t, f_t>
        <<<grid, block_dim, 0, stream>>>(a_values, d_a2l_.data(), nnz_A_, d_Lx_.data(), d_D_.data());
      RAFT_CHECK_CUDA(stream);
    }

    for (i_t l = 0; l < n_levels_; l++) {
      const i_t level_start = level_ptr_[l];
      const i_t level_size  = level_ptr_[l + 1] - level_start;
      ldlt_detail::ldlt_factor_level_kernel<i_t, f_t>
        <<<level_size, block_dim, 0, stream>>>(d_level_cols_.data(),
                                               level_start,
                                               d_Lp_.data(),
                                               d_Li_.data(),
                                               d_rp_ptr_.data(),
                                               d_rp_col_.data(),
                                               d_rp_pos_.data(),
                                               d_Lx_.data(),
                                               d_D_.data(),
                                               d_fail_.data(),
                                               positive_definite_);
      RAFT_CHECK_CUDA(stream);
      if ((l & 63) == 0 && halted()) { return CONCURRENT_HALT_RETURN; }
    }

    i_t fail = d_fail_.value(stream);
    if (halted()) { return CONCURRENT_HALT_RETURN; }
    if (fail != 0) {
      settings_.log.printf("Factorization failed: zero or invalid pivot at column %d\n", fail - 1);
      return -1;
    }

    if (first_factor_) {
      settings_.log.debug("Factorization time          : %.2fs\n", toc(start_numeric));
      first_factor_ = false;
    }
    factorized_ = true;
    return 0;
  }

  // Numeric factorization (host reference implementation).
  // Left-looking by column: for each column j (ascending), apply the updates
  // of all columns k with L(j, k) != 0, then scale by the pivot d_j.
  i_t factorize_values_host()
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
          i_t i   = Li_[p];
          i_t lo  = Lp_[j];
          i_t hi  = Lp_[j + 1];
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

  // Triangular solves on the device with level-scheduled kernels:
  // x = P^T L^{-T} D^{-1} L^{-1} P b. d_b and d_x are device pointers; d_b is
  // left untouched (the work vector carries the intermediate states).
  i_t solve_values_device(const f_t* d_b, f_t* d_x, rmm::cuda_stream_view stream)
  {
    if (!factorized_) {
      settings_.log.printf("Solve called before factorize\n");
      return -1;
    }
    constexpr int block_dim = ldlt_detail::factor_block_dim;
    const int grid_n        = (n_ + block_dim - 1) / block_dim;

    ldlt_detail::ldlt_perm_gather_kernel<i_t, f_t>
      <<<grid_n, block_dim, 0, stream>>>(d_b, d_perm_.data(), n_, d_work_.data());
    RAFT_CHECK_CUDA(stream);

    for (i_t l = 0; l < n_levels_; l++) {
      const i_t level_start = level_ptr_[l];
      const i_t level_size  = level_ptr_[l + 1] - level_start;
      ldlt_detail::ldlt_forward_level_kernel<i_t, f_t>
        <<<level_size, block_dim, 0, stream>>>(d_level_cols_.data(),
                                               level_start,
                                               d_rp_ptr_.data(),
                                               d_rp_col_.data(),
                                               d_rp_pos_.data(),
                                               d_Lx_.data(),
                                               d_work_.data());
      RAFT_CHECK_CUDA(stream);
    }

    ldlt_detail::ldlt_diag_scale_kernel<i_t, f_t>
      <<<grid_n, block_dim, 0, stream>>>(d_D_.data(), n_, d_work_.data());
    RAFT_CHECK_CUDA(stream);

    for (i_t l = n_levels_ - 1; l >= 0; l--) {
      const i_t level_start = level_ptr_[l];
      const i_t level_size  = level_ptr_[l + 1] - level_start;
      ldlt_detail::ldlt_backward_level_kernel<i_t, f_t>
        <<<level_size, block_dim, 0, stream>>>(d_level_cols_.data(),
                                               level_start,
                                               d_Lp_.data(),
                                               d_Li_.data(),
                                               d_Lx_.data(),
                                               d_work_.data());
      RAFT_CHECK_CUDA(stream);
    }

    ldlt_detail::ldlt_perm_scatter_kernel<i_t, f_t>
      <<<grid_n, block_dim, 0, stream>>>(d_work_.data(), d_perm_.data(), n_, d_x);
    RAFT_CHECK_CUDA(stream);
    RAFT_CUDA_TRY(cudaStreamSynchronize(stream));

    if (halted()) { return CONCURRENT_HALT_RETURN; }
    return 0;
  }

  // Triangular solves (host reference implementation):
  // x = P^T L^{-T} D^{-1} L^{-1} P b.
  i_t solve_values_host(const f_t* b, f_t* x)
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
  bool use_host_numeric_;

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

  // Numeric values (host mirror; the device factor lives in d_Lx_/d_D_).
  std::vector<f_t> Lx_;
  std::vector<f_t> D_;
  std::vector<f_t> work_;

  // Host mirror of the analyzed/factorized matrix.
  std::vector<i_t> a_rowptr_host_;
  std::vector<i_t> a_colidx_host_;
  std::vector<f_t> a_values_host_;

  // Device symbolic structures and factor storage.
  rmm::device_uvector<i_t> d_Lp_;
  rmm::device_uvector<i_t> d_Li_;
  rmm::device_uvector<i_t> d_rp_ptr_;
  rmm::device_uvector<i_t> d_rp_col_;
  rmm::device_uvector<i_t> d_rp_pos_;
  rmm::device_uvector<i_t> d_a2l_;
  rmm::device_uvector<i_t> d_level_cols_;
  rmm::device_uvector<i_t> d_perm_;
  rmm::device_uvector<f_t> d_Lx_;
  rmm::device_uvector<f_t> d_D_;
  rmm::device_uvector<f_t> d_a_values_;
  rmm::device_uvector<f_t> d_work_;
  rmm::device_scalar<i_t> d_fail_;
};

}  // namespace cuopt::linear_programming::dual_simplex
