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

#include <raft/core/cublas_macros.hpp>
#include <raft/core/nvtx.hpp>
#include <raft/util/reduction.cuh>
#include <raft/util/warp_constants.hpp>

#include <cublas_v2.h>

#include <rmm/device_scalar.hpp>
#include <rmm/device_uvector.hpp>
#include <rmm/exec_policy.hpp>

#include <thrust/functional.h>
#include <thrust/transform_reduce.h>

#include <amd.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <vector>

namespace cuopt::linear_programming::dual_simplex {

namespace ldlt_detail {

constexpr int factor_block_dim = 128;
constexpr int dense_nb         = 64;  // panel width of the dense-tail factorization
constexpr int ldlt_dense_nb_max = 64;  // compile-time bound for per-thread row buffers

// Replacement pivot for dropped (numerically rank-deficient) columns of an
// SPD system: dividing by it zeroes the column of L and the corresponding
// solution component, so the redundant equation is dropped instead of
// polluting the factor (PCx / LIPSOL treatment of degenerate normal
// equations). Large enough to vanish, small enough that L(i,k)*D(k)*L(j,k)
// stays finite.
constexpr double drop_pivot_value = 1e128;

template <typename f_t>
struct abs_op {
  __host__ __device__ f_t operator()(f_t v) const { return v < f_t(0) ? -v : v; }
};

// Pivot acceptance rule shared by the sparse and dense-tail factor kernels.
// Modes (selected by n_neg, see set_pivot_structure):
//   n_neg <= -2  legacy: unsigned static pivoting at static_pivot_tol.
//   n_neg ==  0  SPD (normal equations): pivots must be positive; a pivot
//                below the tolerance is numerically rank deficient and is
//                dropped (replaced by drop_pivot_value).
//   n_neg  >  0  quasi-definite: original index < n_neg expects a negative
//                pivot (variable block), >= n_neg a positive one (constraint
//                block). The pivot is floored at the block's regularization
//                magnitude with the expected sign; a sign violation is an
//                inertia error and is counted in sign_corrections.
// Returns 1 if the factorization must fail (non-finite pivot or, in legacy
// positive-definite mode, a non-positive pivot). Counters are only updated
// when count is true (one thread per column).
template <typename i_t, typename f_t>
__device__ inline i_t ldlt_pivot_rule(f_t& dj,
                                      i_t orig,
                                      i_t n_neg,
                                      f_t neg_floor,
                                      f_t pos_floor,
                                      f_t static_pivot_tol,
                                      f_t diag0_j,
                                      f_t floor_scale,
                                      bool positive_definite,
                                      bool count,
                                      i_t* static_pivot_count,
                                      i_t* sign_corrections)
{
  if (!isfinite(dj)) { return 1; }
  if (n_neg <= i_t(-2)) {
    if (positive_definite && dj <= f_t(0)) { return 1; }
    if (fabs(dj) < static_pivot_tol) {
      dj = (dj < f_t(0)) ? -static_pivot_tol : static_pivot_tol;
      if (count) { atomicAdd(static_pivot_count, i_t(1)); }
    }
    return 0;
  }
  if (n_neg == i_t(0)) {
    // Detection at the column's own scale: a global max-relative threshold
    // mass-drops legitimate pivots when the diagonal spans many orders of
    // magnitude (late-IPM ADAT diagonals reach 1e14+). Only tiny POSITIVE
    // pivots are dropped (numerical rank deficiency, e.g. redundant rows);
    // negative pivots are cancellation-born on fill-heavy SPD systems
    // (nug08-3rd computes thousands per factorization) and the indefinite
    // factor that keeps them is far more accurate than dropping the rows -
    // they are only floored away from zero.
    const f_t tol_j = f_t(1e-14) * fabs(diag0_j) + f_t(1e-30);
    if (dj >= f_t(0) && dj < tol_j) {
      if (count) { atomicAdd(static_pivot_count, i_t(1)); }
      dj = f_t(drop_pivot_value);
    } else if (dj < f_t(0) && -dj < tol_j) {
      if (count) {
        atomicAdd(sign_corrections, i_t(1));
        atomicAdd(static_pivot_count, i_t(1));
      }
      dj = -tol_j;
    }
    return 0;
  }
  const bool neg  = orig < n_neg;
  const f_t s     = neg ? f_t(-1) : f_t(1);
  const f_t f_blk = (neg ? neg_floor : pos_floor) * floor_scale;
  const f_t fl    = f_blk > f_t(0) ? f_blk : static_pivot_tol;
  if (s * dj < fl) {
    if (count) {
      if (s * dj < f_t(0)) { atomicAdd(sign_corrections, i_t(1)); }
      atomicAdd(static_pivot_count, i_t(1));
    }
    dj = s * fl;
  }
  return 0;
}

// Scatters the values of the analyzed matrix into the factor storage.
// Lx, D and the dense tail must be zeroed beforehand: slots that receive no
// A entry are structural fill. Both halves of a symmetric off-diagonal pair
// map to the same slot and write the same value, so the race is benign.
// Encoding of a2l: [0, nnz_L) sparse slot; >= nnz_L dense slot (offset by
// nnz_L); <= -2 diagonal of permuted index -(v + 2).
template <typename i_t, typename f_t>
__global__ void ldlt_scatter_kernel(
  const f_t* a_values, const int64_t* a2l, i_t nnz_A, i_t nnz_L, f_t* Lx, f_t* S, f_t* D)
{
  const i_t e = static_cast<i_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (e >= nnz_A) { return; }
  const int64_t m = a2l[e];
  if (m < 0) {
    D[-(m + 2)] = a_values[e];
  } else if (m < nnz_L) {
    Lx[m] = a_values[e];
  } else {
    S[m - nnz_L] = a_values[e];
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
                                         i_t* static_pivot_count,
                                         f_t static_pivot_tol,
                                         bool positive_definite,
                                         const i_t* perm,
                                         i_t pivot_n_neg,
                                         f_t pivot_neg_floor,
                                         f_t pivot_pos_floor,
                                         const f_t* diag0,
                                         const f_t* fscale,
                                         i_t* sign_corrections)
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

  f_t dj             = D[j];
  const f_t dj_in    = dj;
  const i_t bad      = ldlt_pivot_rule(dj,
                                  perm[j],
                                  pivot_n_neg,
                                  pivot_neg_floor,
                                  pivot_pos_floor,
                                  static_pivot_tol,
                                  diag0[j],
                                  fscale != nullptr ? fscale[j] * fscale[j] : f_t(1),
                                  positive_definite,
                                  threadIdx.x == 0,
                                  static_pivot_count,
                                  sign_corrections);
  if (bad) {
    if (threadIdx.x == 0) { atomicCAS(fail_flag, i_t(0), j + 1); }
    return;
  }
  if (dj != dj_in && threadIdx.x == 0) { D[j] = dj; }
  for (i_t p = col_beg + static_cast<i_t>(threadIdx.x); p < col_end;
       p += static_cast<i_t>(blockDim.x)) {
    Lx[p] /= dj;
  }
}

// Parallel (non-deterministic) variant of the level factorization, split in
// two phases. The update kernel spreads the contributions of the columns k
// with L(j,k) != 0 over a 2D grid (columns of the level x slices of their row
// patterns) and accumulates with atomics, so narrow levels with long columns
// no longer serialize on a single block. The finalize kernel then applies the
// pivot rules and scales each column.
constexpr int factor_k_slice = 8;

template <typename i_t, typename f_t>
__global__ void ldlt_factor_level_update_kernel(const i_t* level_cols,
                                                i_t level_start,
                                                const i_t* Lp,
                                                const i_t* Li,
                                                const i_t* rp_ptr,
                                                const i_t* rp_col,
                                                const i_t* rp_pos,
                                                f_t* Lx,
                                                f_t* D)
{
  const i_t j   = level_cols[level_start + static_cast<i_t>(blockIdx.x)];
  const i_t beg = rp_ptr[j];
  const i_t end = rp_ptr[j + 1];
  const i_t t0  = beg + static_cast<i_t>(blockIdx.y) * factor_k_slice;
  if (t0 >= end) { return; }
  const i_t t1      = min(end, t0 + factor_k_slice);
  const i_t col_beg = Lp[j];
  const i_t col_end = Lp[j + 1];

  for (i_t t = t0; t < t1; t++) {
    const i_t k    = rp_col[t];
    const i_t p_jk = rp_pos[t];
    const f_t ljk  = Lx[p_jk];
    const f_t c    = ljk * D[k];
    if (threadIdx.x == 0) { atomicAdd(&D[j], -c * ljk); }
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
      atomicAdd(&Lx[lo], -c * Lx[p]);
    }
  }
}

template <typename i_t, typename f_t>
__global__ void ldlt_factor_level_finalize_kernel(const i_t* level_cols,
                                                  i_t level_start,
                                                  const i_t* Lp,
                                                  f_t* Lx,
                                                  f_t* D,
                                                  i_t* fail_flag,
                                                  i_t* static_pivot_count,
                                                  f_t static_pivot_tol,
                                                  bool positive_definite,
                                                  const i_t* perm,
                                                  i_t pivot_n_neg,
                                                  f_t pivot_neg_floor,
                                                  f_t pivot_pos_floor,
                                                  const f_t* diag0,
                                                  const f_t* fscale,
                                                  i_t* sign_corrections)
{
  const i_t j     = level_cols[level_start + static_cast<i_t>(blockIdx.x)];
  f_t dj          = D[j];
  const f_t dj_in = dj;
  const i_t bad   = ldlt_pivot_rule(dj,
                                  perm[j],
                                  pivot_n_neg,
                                  pivot_neg_floor,
                                  pivot_pos_floor,
                                  static_pivot_tol,
                                  diag0[j],
                                  fscale != nullptr ? fscale[j] * fscale[j] : f_t(1),
                                  positive_definite,
                                  threadIdx.x == 0,
                                  static_pivot_count,
                                  sign_corrections);
  if (bad) {
    if (threadIdx.x == 0) { atomicCAS(fail_flag, i_t(0), j + 1); }
    return;
  }
  if (dj != dj_in && threadIdx.x == 0) { D[j] = dj; }
  for (i_t p = Lp[j] + static_cast<i_t>(threadIdx.x); p < Lp[j + 1];
       p += static_cast<i_t>(blockDim.x)) {
    Lx[p] /= dj;
  }
}

// y[k] = b[perm[k]] * (scale ? scale[k] : 1): with the symmetric scaling
// M' = S M S the solve is x = S * M'^{-1} * (S b).
template <typename i_t, typename f_t>
__global__ void ldlt_perm_gather_kernel(
  const f_t* b, const i_t* perm, i_t n, const f_t* scale, f_t* y)
{
  const i_t k = static_cast<i_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (k < n) { y[k] = scale != nullptr ? b[perm[k]] * scale[k] : b[perm[k]]; }
}

// x[perm[k]] = y[k] * (scale ? scale[k] : 1)
template <typename i_t, typename f_t>
__global__ void ldlt_perm_scatter_kernel(
  const f_t* y, const i_t* perm, i_t n, const f_t* scale, f_t* x)
{
  const i_t k = static_cast<i_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (k < n) { x[perm[k]] = scale != nullptr ? y[k] * scale[k] : y[k]; }
}

// Symmetric Jacobi scaling of the assembled (permuted) system:
// scale[j] = 1/sqrt(|diag_j|), entries (i, j) *= scale_i * scale_j. Bounds
// the diagonal to +-1, taming element growth of the pivot-free LDL^T on
// badly ranged quasi-definite KKT systems.
template <typename i_t, typename f_t>
__global__ void ldlt_compute_scale_kernel(const f_t* D, i_t n, f_t* scale)
{
  const i_t j = static_cast<i_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (j >= n) { return; }
  const f_t a = fabs(D[j]);
  scale[j]    = a > f_t(0) ? rsqrt(a) : f_t(1);
}

template <typename i_t, typename f_t>
__global__ void ldlt_apply_scale_kernel(
  const i_t* Lp, const i_t* Li, i_t n, const f_t* scale, f_t* Lx, f_t* D)
{
  const i_t j = static_cast<i_t>(blockIdx.x);
  const f_t sj = scale[j];
  for (i_t p = Lp[j] + static_cast<i_t>(threadIdx.x); p < Lp[j + 1];
       p += static_cast<i_t>(blockDim.x)) {
    Lx[p] *= sj * scale[Li[p]];
  }
  if (threadIdx.x == 0) { D[j] *= sj * sj; }
}

template <typename i_t, typename f_t>
__global__ void ldlt_apply_scale_dense_kernel(
  i_t tail_start, i_t tail_dim, const f_t* scale, f_t* S)
{
  const int64_t t =
    static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (t >= static_cast<int64_t>(tail_dim) * tail_dim) { return; }
  const i_t c = static_cast<i_t>(t / tail_dim);
  const i_t r = static_cast<i_t>(t % tail_dim);
  S[t] *= scale[tail_start + r] * scale[tail_start + c];
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

// ---------------------------------------------------------------------------
// Bundled (fused) level execution. Banded and grid-like problems produce
// elimination trees that are long chains of tiny levels: per-level kernel
// launches are then pure latency (LISWET: 9999 levels of one short column).
// A bundle runs a contiguous range of consecutive light levels inside ONE
// thread block, with __syncthreads() as the level barrier (global-memory
// writes of a block are visible to the block after __syncthreads). Levels
// with a single column use the whole block per column; wider levels assign
// one warp per column. Both paths are deterministic: the k loop is
// sequential per column and the reductions are fixed-order.
// ---------------------------------------------------------------------------

constexpr int bundle_block_dim = 256;

template <typename i_t, typename f_t>
__global__ void ldlt_factor_bundle_kernel(const i_t* level_ptr,
                                          const i_t* level_cols,
                                          i_t level_lo,
                                          i_t level_hi,
                                          const i_t* Lp,
                                          const i_t* Li,
                                          const i_t* rp_ptr,
                                          const i_t* rp_col,
                                          const i_t* rp_pos,
                                          f_t* Lx,
                                          f_t* D,
                                          i_t* fail_flag,
                                          i_t* static_pivot_count,
                                          f_t static_pivot_tol,
                                          bool positive_definite,
                                          const i_t* perm,
                                          i_t pivot_n_neg,
                                          f_t pivot_neg_floor,
                                          f_t pivot_pos_floor,
                                          const f_t* diag0,
                                          const f_t* fscale,
                                          i_t* sign_corrections)
{
  const i_t lane    = static_cast<i_t>(threadIdx.x) & (raft::WarpSize - 1);
  const i_t warp_id = static_cast<i_t>(threadIdx.x) >> raft::WarpSizeLog2;
  const i_t n_warps = static_cast<i_t>(blockDim.x) >> raft::WarpSizeLog2;

  for (i_t l = level_lo; l < level_hi; l++) {
    const i_t lev_beg = level_ptr[l];
    const i_t lev_cnt = level_ptr[l + 1] - lev_beg;
    if (lev_cnt == 1) {
      // Whole block on the single column (the chain case).
      const i_t j       = level_cols[lev_beg];
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
        __syncthreads();
      }
      f_t dj          = D[j];
      const f_t dj_in = dj;
      const i_t bad   = ldlt_pivot_rule(dj,
                                      perm[j],
                                      pivot_n_neg,
                                      pivot_neg_floor,
                                      pivot_pos_floor,
                                      static_pivot_tol,
                                      diag0[j],
                                      fscale != nullptr ? fscale[j] * fscale[j] : f_t(1),
                                      positive_definite,
                                      threadIdx.x == 0,
                                      static_pivot_count,
                                      sign_corrections);
      if (bad) {
        if (threadIdx.x == 0) { atomicCAS(fail_flag, i_t(0), j + 1); }
      } else {
        if (dj != dj_in && threadIdx.x == 0) { D[j] = dj; }
        for (i_t p = col_beg + static_cast<i_t>(threadIdx.x); p < col_end;
             p += static_cast<i_t>(blockDim.x)) {
          Lx[p] /= dj;
        }
      }
    } else {
      // One warp per column of the level.
      for (i_t c0 = warp_id; c0 < lev_cnt; c0 += n_warps) {
        const i_t j       = level_cols[lev_beg + c0];
        const i_t col_beg = Lp[j];
        const i_t col_end = Lp[j + 1];
        for (i_t t = rp_ptr[j]; t < rp_ptr[j + 1]; t++) {
          const i_t k    = rp_col[t];
          const i_t p_jk = rp_pos[t];
          const f_t ljk  = Lx[p_jk];
          const f_t c    = ljk * D[k];
          if (lane == 0) { D[j] -= c * ljk; }
          const i_t k_end = Lp[k + 1];
          for (i_t p = p_jk + 1 + lane; p < k_end; p += raft::WarpSize) {
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
          __syncwarp();
        }
        f_t dj          = D[j];
        const f_t dj_in = dj;
        const i_t bad   = ldlt_pivot_rule(dj,
                                        perm[j],
                                        pivot_n_neg,
                                        pivot_neg_floor,
                                        pivot_pos_floor,
                                        static_pivot_tol,
                                        diag0[j],
                                        fscale != nullptr ? fscale[j] * fscale[j] : f_t(1),
                                        positive_definite,
                                        lane == 0,
                                        static_pivot_count,
                                        sign_corrections);
        if (bad) {
          if (lane == 0) { atomicCAS(fail_flag, i_t(0), j + 1); }
        } else {
          if (dj != dj_in && lane == 0) { D[j] = dj; }
          for (i_t p = col_beg + lane; p < col_end; p += raft::WarpSize) {
            Lx[p] /= dj;
          }
        }
      }
    }
    __syncthreads();  // level barrier
  }
}

// ---------------------------------------------------------------------------
// Chain (width-1 level run) kernels. A run of consecutive single-column
// levels is a sequential dependency chain (banded problems: the whole etree).
// Processing it level-by-level — even fused in one block — pays a global
// round-trip per level. These kernels slide a window of chain_window columns
// over the run: the window's column data lives in shared memory, external
// contributions (columns before the window, already final in global memory)
// are applied warp-parallel, and the unavoidable sequential sweep then runs
// entirely in shared memory. Deterministic: per column, updates apply in row
// pattern order; lanes write disjoint slots.
// ---------------------------------------------------------------------------

constexpr int chain_window  = 512;   // max columns per window
constexpr int chain_col_cap = 32;    // max entries per chain column (analyze enforces)
constexpr int chain_smem    = 1536;  // max window entries (host packs windows to this)

// Binary search for v in s[lo, hi); returns its index (callers rely on the
// factor-pattern guarantee that v is present for scatter targets).
template <typename i_t>
__device__ inline i_t ldlt_smem_search(const i_t* s, i_t lo, i_t hi, i_t v)
{
  while (lo < hi) {
    const i_t mid = lo + (hi - lo) / 2;
    if (s[mid] < v) {
      lo = mid + 1;
    } else {
      hi = mid;
    }
  }
  return lo;
}

// All chain metadata is resolved on the host at analyze time (the pattern is
// static across the IPM iterations): window boundaries (win_ptr, positions
// within the chain), each column's data offset inside its window's shared
// buffer (smoff, window-relative), each row-pattern entry's window slot
// (rp_slot, -1 = external) and each column entry's window slot (col_slot,
// -1 = external), plus per-column op-list offsets (fop_off window-relative
// prefix of in-window row-pattern entries). The kernels then touch global
// memory only in the parallel load/external/store phases; the sequential
// sweep runs entirely in shared memory.
template <typename i_t, typename f_t>
__global__ void ldlt_factor_chain_kernel(const i_t* chain_cols,
                                         const i_t* win_ptr,
                                         i_t n_windows,
                                         const i_t* smoff,    // per chain position
                                         const i_t* fop_off,  // per chain position
                                         const i_t* rp_slot,  // per rp entry (global index)
                                         const i_t* Lp,
                                         const i_t* Li,
                                         const i_t* rp_ptr,
                                         const i_t* rp_col,
                                         const i_t* rp_pos,
                                         f_t* Lx,
                                         f_t* D,
                                         i_t* fail_flag,
                                         i_t* static_pivot_count,
                                         f_t static_pivot_tol,
                                         bool positive_definite,
                                         const i_t* perm,
                                         i_t pivot_n_neg,
                                         f_t pivot_neg_floor,
                                         f_t pivot_pos_floor,
                                         const f_t* diag0,
                                         const f_t* fscale,
                                         i_t* sign_corrections)
{
  __shared__ i_t s_cols[chain_window];
  __shared__ i_t s_smoff[chain_window + 1];
  __shared__ i_t s_fopoff[chain_window + 1];
  __shared__ i_t s_perm[chain_window];
  __shared__ f_t s_d[chain_window];
  // SPD mode needs the assembled diagonal, QD mode the squared floor scale —
  // mutually exclusive, so one auxiliary array serves both.
  __shared__ f_t s_aux[chain_window];
  __shared__ i_t s_li[chain_smem];
  __shared__ f_t s_lx[chain_smem];
  __shared__ i_t s_op_slot[chain_smem];
  __shared__ i_t s_op_q[chain_smem];

  const i_t lane    = static_cast<i_t>(threadIdx.x) & (raft::WarpSize - 1);
  const i_t warp_id = static_cast<i_t>(threadIdx.x) >> raft::WarpSizeLog2;
  const i_t n_warps = static_cast<i_t>(blockDim.x) >> raft::WarpSizeLog2;

  for (i_t w = 0; w < n_windows; w++) {
    const i_t w0 = win_ptr[w];
    const i_t we = win_ptr[w + 1] - w0;
    // Load metadata, column data, diagonals.
    for (i_t c = static_cast<i_t>(threadIdx.x); c < we; c += static_cast<i_t>(blockDim.x)) {
      const i_t j  = chain_cols[w0 + c];
      s_cols[c]    = j;
      s_smoff[c]   = smoff[w0 + c];
      s_fopoff[c]  = fop_off[w0 + c] - fop_off[w0];
      s_perm[c]    = perm[j];
      s_d[c]       = D[j];
      s_aux[c]     = pivot_n_neg == i_t(0)
                       ? diag0[j]
                       : (fscale != nullptr ? fscale[j] * fscale[j] : f_t(1));
      if (c == we - 1) {
        s_smoff[we]  = smoff[w0 + we - 1] + (Lp[j + 1] - Lp[j]);
        s_fopoff[we] = fop_off[w0 + we] - fop_off[w0];
      }
    }
    __syncthreads();
    for (i_t c = warp_id; c < we; c += n_warps) {
      const i_t j   = s_cols[c];
      const i_t len = Lp[j + 1] - Lp[j];
      for (i_t p = lane; p < len; p += raft::WarpSize) {
        s_li[s_smoff[c] + p] = Li[Lp[j] + p];
        s_lx[s_smoff[c] + p] = Lx[Lp[j] + p];
      }
    }
    __syncthreads();
    // External updates (final in global memory) + op-list construction.
    for (i_t c = warp_id; c < we; c += n_warps) {
      const i_t j = s_cols[c];
      i_t n_ops   = 0;
      for (i_t t = rp_ptr[j]; t < rp_ptr[j + 1]; t++) {
        const i_t slot = rp_slot[t];
        if (slot >= 0) {
          if (lane == 0) {
            const i_t k                    = rp_col[t];
            s_op_slot[s_fopoff[c] + n_ops] = slot;
            s_op_q[s_fopoff[c] + n_ops]    = rp_pos[t] - Lp[k] + s_smoff[slot];
          }
          n_ops++;
          continue;
        }
        const i_t k    = rp_col[t];
        const i_t p_jk = rp_pos[t];
        const f_t ljk  = Lx[p_jk];
        const f_t cku  = ljk * D[k];
        if (lane == 0) { s_d[c] -= cku * ljk; }
        const i_t k_end = Lp[k + 1];
        for (i_t p = p_jk + 1 + lane; p < k_end; p += raft::WarpSize) {
          const i_t idx = ldlt_smem_search(s_li, s_smoff[c], s_smoff[c + 1], Li[p]);
          s_lx[idx] -= cku * Lx[p];
        }
        __syncwarp();
      }
    }
    __syncthreads();
    // Sequential in-window sweep (warp 0), entirely in shared memory.
    if (warp_id == 0) {
      for (i_t c = 0; c < we; c++) {
        for (i_t o = s_fopoff[c]; o < s_fopoff[c + 1]; o++) {
          const i_t slot = s_op_slot[o];
          const i_t q    = s_op_q[o];
          const f_t ljk  = s_lx[q];
          const f_t cku  = ljk * s_d[slot];
          if (lane == 0) { s_d[c] -= cku * ljk; }
          for (i_t p = q + 1 + lane; p < s_smoff[slot + 1]; p += raft::WarpSize) {
            const i_t idx = ldlt_smem_search(s_li, s_smoff[c], s_smoff[c + 1], s_li[p]);
            s_lx[idx] -= cku * s_lx[p];
          }
          __syncwarp();
        }
        f_t dj        = s_d[c];
        const i_t bad = ldlt_pivot_rule(dj,
                                        s_perm[c],
                                        pivot_n_neg,
                                        pivot_neg_floor,
                                        pivot_pos_floor,
                                        static_pivot_tol,
                                        pivot_n_neg == i_t(0) ? s_aux[c] : f_t(1),
                                        pivot_n_neg > i_t(0) ? s_aux[c] : f_t(1),
                                        positive_definite,
                                        lane == 0,
                                        static_pivot_count,
                                        sign_corrections);
        if (bad) {
          if (lane == 0) { atomicCAS(fail_flag, i_t(0), s_cols[c] + 1); }
        } else {
          if (lane == 0) { s_d[c] = dj; }
          for (i_t p = s_smoff[c] + lane; p < s_smoff[c + 1]; p += raft::WarpSize) {
            s_lx[p] /= dj;
          }
        }
        __syncwarp();
      }
    }
    __syncthreads();
    // Store the window back.
    for (i_t c = warp_id; c < we; c += n_warps) {
      const i_t j   = s_cols[c];
      const i_t len = s_smoff[c + 1] - s_smoff[c];
      for (i_t p = lane; p < len; p += raft::WarpSize) {
        Lx[Lp[j] + p] = s_lx[s_smoff[c] + p];
      }
      if (lane == 0) { D[j] = s_d[c]; }
    }
    __syncthreads();
  }
}

// Forward substitution over a chain: per window, external row entries
// (rp_slot == -1) are reduced thread-parallel against final global y,
// internal entries become (slot, value) ops, and the sequential sweep runs in
// shared memory.
template <typename i_t, typename f_t>
__global__ void ldlt_forward_chain_kernel(const i_t* chain_cols,
                                          const i_t* win_ptr,
                                          i_t n_windows,
                                          const i_t* fop_off,  // per chain position
                                          const i_t* rp_slot,  // per rp entry
                                          const i_t* rp_ptr,
                                          const i_t* rp_col,
                                          const i_t* rp_pos,
                                          const f_t* Lx,
                                          f_t* y)
{
  __shared__ i_t s_cols[chain_window];
  __shared__ f_t s_y[chain_window];
  __shared__ f_t s_ext[chain_window];
  __shared__ i_t s_op_off[chain_window + 1];
  __shared__ i_t s_op_slot[chain_smem];
  __shared__ f_t s_op_val[chain_smem];

  for (i_t w = 0; w < n_windows; w++) {
    const i_t w0 = win_ptr[w];
    const i_t we = win_ptr[w + 1] - w0;
    for (i_t c = static_cast<i_t>(threadIdx.x); c < we; c += static_cast<i_t>(blockDim.x)) {
      const i_t j = chain_cols[w0 + c];
      s_cols[c]   = j;
      s_y[c]      = y[j];
      s_op_off[c] = fop_off[w0 + c] - fop_off[w0];
      if (c == we - 1) { s_op_off[we] = fop_off[w0 + we] - fop_off[w0]; }
    }
    __syncthreads();
    // Split row entries: external -> partial sums now; internal -> op list.
    for (i_t c = static_cast<i_t>(threadIdx.x); c < we; c += static_cast<i_t>(blockDim.x)) {
      const i_t j = s_cols[c];
      f_t partial = f_t(0);
      i_t n_in    = 0;
      for (i_t t = rp_ptr[j]; t < rp_ptr[j + 1]; t++) {
        const i_t slot = rp_slot[t];
        if (slot >= 0) {
          s_op_slot[s_op_off[c] + n_in] = slot;
          s_op_val[s_op_off[c] + n_in]  = Lx[rp_pos[t]];
          n_in++;
        } else {
          partial += Lx[rp_pos[t]] * y[rp_col[t]];
        }
      }
      s_ext[c] = partial;
    }
    __syncthreads();
    // Sequential sweep in shared memory.
    if (threadIdx.x == 0) {
      for (i_t c = 0; c < we; c++) {
        f_t v = s_y[c] - s_ext[c];
        for (i_t o = s_op_off[c]; o < s_op_off[c + 1]; o++) {
          v -= s_op_val[o] * s_y[s_op_slot[o]];
        }
        s_y[c] = v;
      }
    }
    __syncthreads();
    for (i_t c = static_cast<i_t>(threadIdx.x); c < we; c += static_cast<i_t>(blockDim.x)) {
      y[s_cols[c]] = s_y[c];
    }
    __syncthreads();
  }
}

// Backward substitution over a chain (windows and the in-window sweep run in
// reverse). Column entries point to later columns: in-window ones (col_slot
// >= 0) become (slot, value) ops against s_y; the rest read final global y.
template <typename i_t, typename f_t>
__global__ void ldlt_backward_chain_kernel(const i_t* chain_cols,
                                           const i_t* win_ptr,
                                           i_t n_windows,
                                           const i_t* bop_off,   // per chain position
                                           const i_t* col_slot,  // per column entry
                                           const i_t* Lp,
                                           const i_t* Li,
                                           const f_t* Lx,
                                           f_t* y)
{
  __shared__ i_t s_cols[chain_window];
  __shared__ f_t s_y[chain_window];
  __shared__ f_t s_ext[chain_window];
  __shared__ i_t s_op_off[chain_window + 1];
  __shared__ i_t s_op_slot[chain_smem];
  __shared__ f_t s_op_val[chain_smem];

  for (i_t w = n_windows - 1; w >= 0; w--) {
    const i_t w0 = win_ptr[w];
    const i_t we = win_ptr[w + 1] - w0;
    for (i_t c = static_cast<i_t>(threadIdx.x); c < we; c += static_cast<i_t>(blockDim.x)) {
      const i_t j = chain_cols[w0 + c];
      s_cols[c]   = j;
      s_y[c]      = y[j];
      s_op_off[c] = bop_off[w0 + c] - bop_off[w0];
      if (c == we - 1) { s_op_off[we] = bop_off[w0 + we] - bop_off[w0]; }
    }
    __syncthreads();
    for (i_t c = static_cast<i_t>(threadIdx.x); c < we; c += static_cast<i_t>(blockDim.x)) {
      const i_t j = s_cols[c];
      f_t partial = f_t(0);
      i_t n_in    = 0;
      for (i_t p = Lp[j]; p < Lp[j + 1]; p++) {
        const i_t slot = col_slot[p];
        if (slot >= 0) {
          s_op_slot[s_op_off[c] + n_in] = slot;
          s_op_val[s_op_off[c] + n_in]  = Lx[p];
          n_in++;
        } else {
          partial += Lx[p] * y[Li[p]];
        }
      }
      s_ext[c] = partial;
    }
    __syncthreads();
    if (threadIdx.x == 0) {
      for (i_t c = we - 1; c >= 0; c--) {
        f_t v = s_y[c] - s_ext[c];
        for (i_t o = s_op_off[c]; o < s_op_off[c + 1]; o++) {
          v -= s_op_val[o] * s_y[s_op_slot[o]];
        }
        s_y[c] = v;
      }
    }
    __syncthreads();
    for (i_t c = static_cast<i_t>(threadIdx.x); c < we; c += static_cast<i_t>(blockDim.x)) {
      y[s_cols[c]] = s_y[c];
    }
    __syncthreads();
  }
}

// Forward substitution over a bundle of levels: y[j] -= row_j(L) . y.
template <typename i_t, typename f_t>
__global__ void ldlt_forward_bundle_kernel(const i_t* level_ptr,
                                           const i_t* level_cols,
                                           i_t level_lo,
                                           i_t level_hi,
                                           const i_t* rp_ptr,
                                           const i_t* rp_col,
                                           const i_t* rp_pos,
                                           const f_t* Lx,
                                           f_t* y)
{
  __shared__ f_t sdata[bundle_block_dim];
  const i_t lane    = static_cast<i_t>(threadIdx.x) & (raft::WarpSize - 1);
  const i_t warp_id = static_cast<i_t>(threadIdx.x) >> raft::WarpSizeLog2;
  const i_t n_warps = static_cast<i_t>(blockDim.x) >> raft::WarpSizeLog2;

  for (i_t l = level_lo; l < level_hi; l++) {
    const i_t lev_beg = level_ptr[l];
    const i_t lev_cnt = level_ptr[l + 1] - lev_beg;
    if (lev_cnt == 1) {
      const i_t j = level_cols[lev_beg];
      f_t partial = f_t(0);
      for (i_t t = rp_ptr[j] + static_cast<i_t>(threadIdx.x); t < rp_ptr[j + 1];
           t += static_cast<i_t>(blockDim.x)) {
        partial += Lx[rp_pos[t]] * y[rp_col[t]];
      }
      const f_t s = ldlt_block_reduce(partial, sdata);
      if (threadIdx.x == 0) { y[j] -= s; }
    } else {
      for (i_t c0 = warp_id; c0 < lev_cnt; c0 += n_warps) {
        const i_t j = level_cols[lev_beg + c0];
        f_t partial = f_t(0);
        for (i_t t = rp_ptr[j] + lane; t < rp_ptr[j + 1]; t += raft::WarpSize) {
          partial += Lx[rp_pos[t]] * y[rp_col[t]];
        }
        const f_t s = raft::warpReduce(partial);
        if (lane == 0) { y[j] -= s; }
      }
    }
    __syncthreads();
  }
}

// Backward substitution over a bundle of levels (levels run in reverse):
// y[j] -= col_j(L) . y.
template <typename i_t, typename f_t>
__global__ void ldlt_backward_bundle_kernel(const i_t* level_ptr,
                                            const i_t* level_cols,
                                            i_t level_lo,
                                            i_t level_hi,
                                            const i_t* Lp,
                                            const i_t* Li,
                                            const f_t* Lx,
                                            f_t* y)
{
  __shared__ f_t sdata[bundle_block_dim];
  const i_t lane    = static_cast<i_t>(threadIdx.x) & (raft::WarpSize - 1);
  const i_t warp_id = static_cast<i_t>(threadIdx.x) >> raft::WarpSizeLog2;
  const i_t n_warps = static_cast<i_t>(blockDim.x) >> raft::WarpSizeLog2;

  for (i_t l = level_hi - 1; l >= level_lo; l--) {
    const i_t lev_beg = level_ptr[l];
    const i_t lev_cnt = level_ptr[l + 1] - lev_beg;
    if (lev_cnt == 1) {
      const i_t j = level_cols[lev_beg];
      f_t partial = f_t(0);
      for (i_t p = Lp[j] + static_cast<i_t>(threadIdx.x); p < Lp[j + 1];
           p += static_cast<i_t>(blockDim.x)) {
        partial += Lx[p] * y[Li[p]];
      }
      const f_t s = ldlt_block_reduce(partial, sdata);
      if (threadIdx.x == 0) { y[j] -= s; }
    } else {
      for (i_t c0 = warp_id; c0 < lev_cnt; c0 += n_warps) {
        const i_t j = level_cols[lev_beg + c0];
        f_t partial = f_t(0);
        for (i_t p = Lp[j] + lane; p < Lp[j + 1]; p += raft::WarpSize) {
          partial += Lx[p] * y[Li[p]];
        }
        const f_t s = raft::warpReduce(partial);
        if (lane == 0) { y[j] -= s; }
      }
    }
    __syncthreads();
  }
}

// ---------------------------------------------------------------------------
// Dense-tail kernels. The trailing columns of an AMD-ordered factor are nearly
// dense and form the long sequential path of the elimination tree; they are
// factored as one dense LDL^T block (column-major, dimension tail_dim, column
// j of the tail is global column tail_start + j). The diagonal stays in the
// global D vector; the dense block holds L with an implicit unit diagonal.
// ---------------------------------------------------------------------------

// Accumulates the Schur complement of "light" head columns (few tail
// entries) into the dense tail: for every listed column k and every pair of
// tail entries (i, j) of that column, S(i, j) -= L(i,k) * d_k * L(j,k).
// One block per listed column; threads stride over the lower-triangular
// pairs. Uses atomics (different columns hit the same S entry), so this path
// is not bitwise deterministic; the dense tail is disabled when the
// deterministic mode is requested.
template <typename i_t, typename f_t>
__global__ void ldlt_schur_pairs_kernel(const i_t* cols,
                                        const i_t* Lp,
                                        const i_t* Li,
                                        const i_t* tail_seg,
                                        const f_t* Lx,
                                        const f_t* D,
                                        i_t tail_start,
                                        i_t tail_dim,
                                        f_t* S,
                                        f_t* D_out)
{
  const i_t k   = cols[blockIdx.x];
  const i_t beg = tail_seg[k];
  const i_t end = Lp[k + 1];
  const i_t m   = end - beg;
  if (m <= 0) { return; }
  const f_t dk = D[k];
  // pairs (a, b) with 0 <= b <= a < m: row index Li[beg+a] >= col Li[beg+b]
  const int64_t n_pairs = static_cast<int64_t>(m) * (m + 1) / 2;
  for (int64_t t = threadIdx.x; t < n_pairs; t += blockDim.x) {
    // invert triangular index: a = floor((sqrt(8t+1)-1)/2), b = t - a(a+1)/2
    int64_t a = static_cast<int64_t>((sqrt(8.0 * static_cast<double>(t) + 1.0) - 1.0) * 0.5);
    while ((a + 1) * (a + 2) / 2 <= t) {
      a++;
    }
    while (a * (a + 1) / 2 > t) {
      a--;
    }
    const int64_t b = t - a * (a + 1) / 2;
    const f_t lik   = Lx[beg + a];
    const f_t ljk   = Lx[beg + b];
    const f_t v     = -lik * dk * ljk;
    if (a == b) {
      atomicAdd(&D_out[Li[beg + a]], v);
    } else {
      const i_t gi = Li[beg + a] - tail_start;
      const i_t gj = Li[beg + b] - tail_start;
      atomicAdd(&S[static_cast<int64_t>(gj) * tail_dim + gi], v);
    }
  }
}

// Gathers a chunk of "heavy" head columns (many tail entries) into dense
// panels V (raw values) and W (scaled by d_k), so their Schur contribution
// becomes one DGEMM: S -= V * W^T. One block per column of the chunk.
// V and W must be zeroed beforehand.
template <typename i_t, typename f_t>
__global__ void ldlt_schur_gather_kernel(const i_t* heavy_cols,
                                         i_t chunk_beg,
                                         const i_t* Lp,
                                         const i_t* Li,
                                         const i_t* tail_seg,
                                         const f_t* Lx,
                                         const f_t* D,
                                         i_t tail_start,
                                         i_t tail_dim,
                                         f_t* V,
                                         f_t* W)
{
  const i_t c      = static_cast<i_t>(blockIdx.x);
  const i_t k      = heavy_cols[chunk_beg + c];
  const i_t beg    = tail_seg[k];
  const i_t end    = Lp[k + 1];
  const f_t dk     = D[k];
  const int64_t cb = static_cast<int64_t>(c) * tail_dim;
  for (i_t p = beg + static_cast<i_t>(threadIdx.x); p < end; p += static_cast<i_t>(blockDim.x)) {
    const i_t r = Li[p] - tail_start;
    const f_t v = Lx[p];
    V[cb + r]   = v;
    W[cb + r]   = v * dk;
  }
}

// Factors the nb x nb diagonal block starting at (p0, p0) of the dense tail
// in place (single block; the block is small). Applies the same pivot rules
// as the sparse kernel. The diagonal goes to D[tail_start + p0 + c]; the
// block's stored diagonal is left as is (consumers use unit-diagonal views).
template <typename i_t, typename f_t>
__global__ void ldlt_dense_diag_kernel(f_t* S,
                                       i_t tail_dim,
                                       i_t tail_start,
                                       i_t p0,
                                       i_t nb,
                                       f_t* D,
                                       i_t* fail_flag,
                                       i_t* static_pivot_count,
                                       f_t static_pivot_tol,
                                       bool positive_definite,
                                       const i_t* perm,
                                       i_t pivot_n_neg,
                                       f_t pivot_neg_floor,
                                       f_t pivot_pos_floor,
                                       const f_t* diag0,
                                       const f_t* fscale,
                                       i_t* sign_corrections)
{
  for (i_t c = 0; c < nb; c++) {
    const int64_t col_c = static_cast<int64_t>(p0 + c) * tail_dim;
    f_t dc              = S[col_c + p0 + c] + D[tail_start + p0 + c];
    // D held the assembled diagonal; S's diagonal entry holds within-panel
    // updates accumulated below. Fold and reset so the load is idempotent.
    if (threadIdx.x == 0) {
      S[col_c + p0 + c] = f_t(0);
      const i_t bad     = ldlt_pivot_rule(dc,
                                      perm[tail_start + p0 + c],
                                      pivot_n_neg,
                                      pivot_neg_floor,
                                      pivot_pos_floor,
                                      static_pivot_tol,
                                      diag0[tail_start + p0 + c],
                                      fscale != nullptr ? fscale[tail_start + p0 + c] * fscale[tail_start + p0 + c]
                                                        : f_t(1),
                                      positive_definite,
                                      true,
                                      static_pivot_count,
                                      sign_corrections);
      if (bad) { atomicCAS(fail_flag, i_t(0), tail_start + p0 + c + 1); }
      D[tail_start + p0 + c] = dc;
    }
    __syncthreads();
    dc = D[tail_start + p0 + c];
    // scale column c below the diagonal (within the diagonal block only)
    for (i_t r = c + 1 + static_cast<i_t>(threadIdx.x); r < nb;
         r += static_cast<i_t>(blockDim.x)) {
      S[col_c + p0 + r] /= dc;
    }
    __syncthreads();
    // right-looking rank-1 update of the remaining panel columns: the target
    // entries (cc, r) with c < cc <= r < nb are independent; flatten the
    // triangle (diagonal entries included, folded later) across the block.
    const i_t m           = nb - c - 1;  // remaining columns/rows
    const int64_t n_elems = static_cast<int64_t>(m) * (m + 1) / 2;
    for (int64_t t = threadIdx.x; t < n_elems; t += blockDim.x) {
      int64_t a = static_cast<int64_t>((sqrt(8.0 * static_cast<double>(t) + 1.0) - 1.0) * 0.5);
      while ((a + 1) * (a + 2) / 2 <= t) {
        a++;
      }
      while (a * (a + 1) / 2 > t) {
        a--;
      }
      const int64_t b      = t - a * (a + 1) / 2;  // 0 <= b <= a < m
      const i_t cc         = c + 1 + static_cast<i_t>(b);
      const i_t r          = c + 1 + static_cast<i_t>(a);
      const int64_t col_cc = static_cast<int64_t>(p0 + cc) * tail_dim;
      S[col_cc + p0 + r] -= S[col_c + p0 + cc] * dc * S[col_c + p0 + r];
    }
    __syncthreads();
  }
}

// Solves the panel below a factored diagonal block:
// L(r, p0+c) = (S(r, p0+c) - sum_{cc<c} L(r, p0+cc) d_cc L(p0+c, p0+cc)) / d_c
// One thread per row r in (p0+nb, tail_dim).
template <typename i_t, typename f_t>
__global__ void ldlt_dense_panel_solve_kernel(
  f_t* S, i_t tail_dim, i_t tail_start, i_t p0, i_t nb, const f_t* D)
{
  // Cache the (unit-lower) diagonal block, pre-scaled by D, plus 1/D in
  // shared memory: thread r then only touches its own row in global memory.
  extern __shared__ unsigned char smem_raw[];
  f_t* LD     = reinterpret_cast<f_t*>(smem_raw);  // nb x nb, LD(j, cc) = L(j,cc) * d_cc
  f_t* inv_d  = LD + static_cast<int64_t>(nb) * nb;
  for (i_t t = threadIdx.x; t < nb * nb; t += blockDim.x) {
    const i_t cc = t / nb;
    const i_t jj = t % nb;
    LD[t] = (jj > cc)
              ? S[static_cast<int64_t>(p0 + cc) * tail_dim + (p0 + jj)] * D[tail_start + p0 + cc]
              : f_t(0);
  }
  for (i_t t = threadIdx.x; t < nb; t += blockDim.x) {
    inv_d[t] = f_t(1) / D[tail_start + p0 + t];
  }
  __syncthreads();

  const i_t r = p0 + nb + static_cast<i_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (r >= tail_dim) { return; }
  f_t row[ldlt_dense_nb_max];
  for (i_t c = 0; c < nb; c++) {
    row[c] = S[static_cast<int64_t>(p0 + c) * tail_dim + r];
  }
  for (i_t c = 0; c < nb; c++) {
    f_t v = row[c];
    for (i_t cc = 0; cc < c; cc++) {
      v -= row[cc] * LD[static_cast<int64_t>(cc) * nb + c];
    }
    v *= inv_d[c];
    row[c] = v;
  }
  for (i_t c = 0; c < nb; c++) {
    S[static_cast<int64_t>(p0 + c) * tail_dim + r] = row[c];
  }
}

// M = (L(p0:p0+nb, 0:p0) * diag(D))^T: the (p0 x nb) right factor of the
// left-looking panel update GEMM.
template <typename i_t, typename f_t>
__global__ void ldlt_dense_panel_slab_kernel(
  const f_t* S, i_t tail_dim, i_t tail_start, i_t p0, i_t nb, const f_t* D, f_t* M)
{
  const int64_t t = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (t >= static_cast<int64_t>(p0) * nb) { return; }
  const i_t j = static_cast<i_t>(t / p0);  // panel column 0..nb
  const i_t c = static_cast<i_t>(t % p0);  // previous column 0..p0
  M[static_cast<int64_t>(j) * p0 + c] =
    S[static_cast<int64_t>(c) * tail_dim + (p0 + j)] * D[tail_start + c];
}

// Forward-solve coupling into the tail: for every tail row j, subtract the
// contributions of its head entries: y[j] -= sum L(j,k) y[k] over k < tail
// start. One block per tail row; deterministic fixed-order reduction.
template <typename i_t, typename f_t>
__global__ void ldlt_forward_tail_couple_kernel(i_t tail_start,
                                                const i_t* rp_ptr,
                                                const i_t* rp_head_end,
                                                const i_t* rp_col,
                                                const i_t* rp_pos,
                                                const f_t* Lx,
                                                f_t* y)
{
  __shared__ f_t sdata[factor_block_dim];
  const i_t j   = tail_start + static_cast<i_t>(blockIdx.x);
  const i_t beg = rp_ptr[j];
  const i_t end = rp_head_end[j];
  f_t partial   = f_t(0);
  for (i_t t = beg + static_cast<i_t>(threadIdx.x); t < end; t += static_cast<i_t>(blockDim.x)) {
    partial += Lx[rp_pos[t]] * y[rp_col[t]];
  }
  const f_t s = ldlt_block_reduce(partial, sdata);
  if (threadIdx.x == 0) { y[j] -= s; }
}

// Postorder of the elimination tree (iterative DFS, children in index order).
template <typename i_t>
inline std::vector<i_t> ldlt_etree_postorder(const std::vector<i_t>& parent)
{
  const i_t n = static_cast<i_t>(parent.size());
  // Build child lists (head/next).
  std::vector<i_t> head(n, -1), next(n, -1), post(n), stack(n);
  for (i_t j = n - 1; j >= 0; j--) {
    const i_t p = parent[j];
    if (p != -1) {
      next[j] = head[p];
      head[p] = j;
    }
  }
  i_t k = 0;
  for (i_t root = 0; root < n; root++) {
    if (parent[root] != -1) { continue; }
    i_t top      = 0;
    stack[top]   = root;
    while (top >= 0) {
      const i_t j = stack[top];
      const i_t c = head[j];
      if (c == -1) {
        post[k++] = j;
        top--;
      } else {
        head[j]    = next[c];  // pop child from list
        stack[++top] = c;
      }
    }
  }
  return post;
}

// Exact column counts of the Cholesky factor (Gilbert-Ng-Peyton, the
// cs_counts algorithm from Davis, "Direct Methods for Sparse Linear
// Systems"): colcount[j] = |{i >= j : L(i, j) != 0}| including the diagonal,
// in O(nnz * alpha(n)) without forming the patterns. lc[j] must hold the
// strict lower pattern of (permuted) column j, i.e. { i > j : A(i, j) != 0 }.
template <typename i_t>
inline std::vector<i_t> ldlt_column_counts(const std::vector<std::vector<i_t>>& lc,
                                           const std::vector<i_t>& parent,
                                           const std::vector<i_t>& post)
{
  const i_t n = static_cast<i_t>(parent.size());
  std::vector<i_t> colcount(n), first(n, -1), maxfirst(n, -1), prevleaf(n, -1), ancestor(n);
  // first[j] = postorder index of the first descendant of j; a node is a leaf
  // of the etree iff it is its own first descendant.
  for (i_t k = 0; k < n; k++) {
    i_t j       = post[k];
    colcount[j] = (first[j] == -1) ? 1 : 0;  // delta init: leaves count themselves
    for (; j != -1 && first[j] == -1; j = parent[j]) {
      first[j] = k;
    }
  }
  for (i_t i = 0; i < n; i++) {
    ancestor[i] = i;
  }
  for (i_t k = 0; k < n; k++) {
    const i_t j = post[k];
    if (parent[j] != -1) { colcount[parent[j]]--; }  // j contributes its subtree once
    for (const i_t i : lc[j]) {
      // j is a new leaf of the i-th row subtree?
      if (first[j] > maxfirst[i]) {
        maxfirst[i] = first[j];
        colcount[j]++;
        const i_t pl = prevleaf[i];
        if (pl != -1) {
          // Subtract the overlap at the least common ancestor of the previous
          // leaf and j (union-find with path compression).
          i_t q = pl;
          while (q != ancestor[q]) {
            q = ancestor[q];
          }
          for (i_t s = pl; s != q;) {
            const i_t s_next = ancestor[s];
            ancestor[s]      = q;
            s                = s_next;
          }
          colcount[q]--;
        }
        prevleaf[i] = j;
      }
    }
    if (parent[j] != -1) { ancestor[j] = parent[j]; }
  }
  // Accumulate the deltas up the tree (children before parents).
  for (i_t k = 0; k < n; k++) {
    const i_t j = post[k];
    if (parent[j] != -1) { colcount[parent[j]] += colcount[j]; }
  }
  return colcount;
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
      stats_enabled_(std::getenv("CUOPT_LDLT_STATS") != nullptr),
      d_Lp_(0, handle_ptr->get_stream()),
      d_Li_(0, handle_ptr->get_stream()),
      d_rp_ptr_(0, handle_ptr->get_stream()),
      d_rp_col_(0, handle_ptr->get_stream()),
      d_rp_pos_(0, handle_ptr->get_stream()),
      d_a2l_(0, handle_ptr->get_stream()),
      d_level_cols_(0, handle_ptr->get_stream()),
      d_head_level_cols_(0, handle_ptr->get_stream()),
      d_head_level_ptr_(0, handle_ptr->get_stream()),
      d_tail_seg_(0, handle_ptr->get_stream()),
      d_rp_head_end_(0, handle_ptr->get_stream()),
      d_chain_win_(0, handle_ptr->get_stream()),
      d_chain_smoff_(0, handle_ptr->get_stream()),
      d_chain_fop_off_(0, handle_ptr->get_stream()),
      d_chain_bop_off_(0, handle_ptr->get_stream()),
      d_chain_rp_slot_(0, handle_ptr->get_stream()),
      d_chain_col_slot_(0, handle_ptr->get_stream()),
      d_schur_light_(0, handle_ptr->get_stream()),
      d_schur_heavy_(0, handle_ptr->get_stream()),
      d_dense_(0, handle_ptr->get_stream()),
      d_panelV_(0, handle_ptr->get_stream()),
      d_panelW_(0, handle_ptr->get_stream()),
      d_perm_(0, handle_ptr->get_stream()),
      d_Lx_(0, handle_ptr->get_stream()),
      d_D_(0, handle_ptr->get_stream()),
      d_a_values_(0, handle_ptr->get_stream()),
      d_work_(0, handle_ptr->get_stream()),
      d_diag0_(0, handle_ptr->get_stream()),
      d_scale_(0, handle_ptr->get_stream()),
      d_solve_b_(0, handle_ptr->get_stream()),
      d_solve_x_(0, handle_ptr->get_stream()),
      d_fail_(handle_ptr->get_stream()),
      d_static_pivots_(handle_ptr->get_stream()),
      d_sign_corrections_(handle_ptr->get_stream())
  {
    settings_.log.printf("Sparse LDLT solver          : cuOpt built-in (%s numeric)\n",
                         use_host_numeric_ ? "host" : "device");
  }

  ~sparse_cholesky_ldlt_t() override
  {
    if (solve_graph_valid_) { cudaGraphExecDestroy(solve_graph_); }
    if (factor_graph_valid_) { cudaGraphExecDestroy(factor_graph_); }
    if (std::getenv("CUOPT_LDLT_STATS") != nullptr) {
      settings_.log.printf(
        "LDLT stats: factorize calls=%d total=%.2fs | solve calls=%d total=%.2fs\n",
        n_factor_calls_,
        factor_time_,
        n_solve_calls_,
        solve_time_);
    }
  }

  void set_positive_definite(bool positive_definite) override
  {
    positive_definite_ = positive_definite;
  }

  // Structured pivot handling (see ldlt_detail::ldlt_pivot_rule). n_negative
  // == 0 selects the SPD drop rule (normal equations); n_negative > 0 selects
  // the quasi-definite signed floors, where original indices < n_negative
  // (the variable block of the augmented KKT system) expect negative pivots
  // with magnitude >= negative_floor and the rest expect positive pivots
  // >= positive_floor. Callable between factorizations (floors track the
  // caller's adaptive regularization).
  void set_pivot_structure(i_t n_negative, f_t negative_floor, f_t positive_floor) override
  {
    pivot_n_neg_     = n_negative;
    pivot_neg_floor_ = negative_floor;
    pivot_pos_floor_ = positive_floor;
  }

  // Number of expected-sign violations corrected during the last factorize.
  i_t pivot_corrections() const override { return last_sign_corrections_; }

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

  // Picks the dense trailing block from the exact factor column counts: the
  // largest trailing block whose true fill density clears a threshold (the
  // genuinely dense top of the elimination tree, e.g. fill-heavy AMD factors).
  // Sparse tops — banded/grid problems whose chains used to be absorbed here
  // for scheduling reasons — are left to the bundled level kernels instead of
  // being densified. Capped by device memory.
  void choose_tail_density()
  {
    tail_dim_   = 0;
    tail_start_ = n_;
    if (use_host_numeric_) { return; }
    if (settings_.cudss_deterministic) { return; }  // Schur assembly uses atomics
    const char* env = std::getenv("CUOPT_LDLT_TAIL");
    i_t forced      = env != nullptr ? std::atoi(env) : -1;
    if (env != nullptr && forced <= 0) { return; }

    size_t free_b = 0, total_b = 0;
    if (cudaMemGetInfo(&free_b, &total_b) != cudaSuccess) { return; }
    const i_t d_mem = static_cast<i_t>(std::sqrt(0.30 * static_cast<double>(free_b) / 8.0));
    const i_t d_cap = std::min<i_t>(n_, d_mem);
    if (forced > 0) {
      tail_dim_   = std::min(forced, d_cap);
      tail_start_ = n_ - tail_dim_;
      return;
    }
    if (n_ < 256) { return; }

    const char* tau_env = std::getenv("CUOPT_LDLT_TAIL_DENSITY");
    const double tau    = tau_env != nullptr ? std::atof(tau_env) : 0.5;

    // Marginal-density scan: extend the block downward only while the columns
    // being added are themselves dense (windowed). An average-density rule
    // lets a dense core subsidize thousands of mediocre columns, and the
    // Schur assembly then pays 2*d^2 flops per coupled head column
    // (CVXQP1_L grew a 3117-wide tail with ~2e11 Schur flops per factorize).
    constexpr i_t mw   = 64;  // marginal window
    int64_t win_nnz    = 0;
    int64_t win_area   = 0;
    i_t best_t         = n_;
    for (i_t t = n_ - 1; t >= 0 && n_ - t <= d_cap; t--) {
      win_nnz += col_counts_[t] - 1;
      win_area += n_ - t - 1;
      if (n_ - t > mw) {
        const i_t drop = t + mw;  // column leaving the window
        win_nnz -= col_counts_[drop] - 1;
        win_area -= n_ - drop - 1;
      }
      const i_t d = n_ - t;
      if (win_area > 0 && static_cast<double>(win_nnz) < tau * static_cast<double>(win_area)) {
        break;  // the columns being added are no longer dense
      }
      if (d >= 128) { best_t = t; }
    }
    if (best_t < n_) {
      tail_start_ = best_t;
      tail_dim_   = n_ - best_t;
    }
  }

  // Groups consecutive light head levels into bundled segments executed by a
  // single kernel launch (see ldlt_factor_bundle_kernel). Heavy levels stay
  // on the per-level grid kernels. Separate plans for the factorization and
  // the two triangular solves, since their per-level work differs. A bundle
  // of width-1 micro levels (chains) runs on a single warp.
  void build_level_bundles()
  {
    factor_segs_.clear();
    fwd_segs_.clear();
    bwd_segs_.clear();
    if (n_head_levels_ == 0) { return; }

    constexpr i_t max_bundle_cols     = 64;
    constexpr int64_t max_bundle_work = 4096;
    // One block executes a bundle serially: cap the per-launch total so a
    // long run of near-threshold levels does not become a millisecond-long
    // single-block kernel.
    constexpr int64_t max_seg_work = 16384;
    constexpr i_t chain_min_levels = 64;  // below one window a plain bundle is fine

    std::vector<int64_t> factor_work(n_head_levels_, 0);
    std::vector<int64_t> fwd_work(n_head_levels_, 0);
    std::vector<int64_t> bwd_work(n_head_levels_, 0);
    std::vector<i_t> width(n_head_levels_, 0);
    std::vector<i_t> max_col(n_head_levels_, 0);
    std::vector<char> chain_ok(n_head_levels_, 0);
    for (i_t l = 0; l < n_head_levels_; l++) {
      width[l] = head_level_ptr_[l + 1] - head_level_ptr_[l];
      for (i_t t = head_level_ptr_[l]; t < head_level_ptr_[l + 1]; t++) {
        const i_t j      = head_level_cols_[t];
        const i_t rp_len = rp_ptr_[j + 1] - rp_ptr_[j];
        const i_t c_len  = Lp_[j + 1] - Lp_[j];
        int64_t scatter  = 0;
        for (i_t u = rp_ptr_[j]; u < rp_ptr_[j + 1]; u++) {
          scatter += Lp_[rp_col_[u] + 1] - rp_pos_[u] - 1;
        }
        // Each scatter target is a binary search over the (global-memory)
        // column pattern inside a single block: weight by the search depth so
        // long-column levels price themselves out of bundles.
        i_t search_depth = 1;
        while ((i_t(1) << search_depth) < c_len) {
          search_depth++;
        }
        factor_work[l] += rp_len + scatter * search_depth + c_len;
        fwd_work[l] += rp_len;
        bwd_work[l] += c_len;
        max_col[l] = std::max(max_col[l], c_len);
        if (width[l] == 1) {
          chain_ok[l] = rp_len <= ldlt_detail::chain_col_cap && c_len <= ldlt_detail::chain_col_cap;
        }
      }
    }
    // Length of the chain-eligible run starting at each level.
    std::vector<i_t> chain_run(n_head_levels_ + 1, 0);
    for (i_t l = n_head_levels_ - 1; l >= 0; l--) {
      chain_run[l] = chain_ok[l] ? chain_run[l + 1] + 1 : 0;
    }

    // The factor bundle scatters through per-entry binary searches over the
    // target column in global memory: long columns cost ~log2(len) global
    // round-trips per entry inside a single block (no latency hiding), so
    // factor bundles additionally require short columns. The solve bundles
    // read their entries directly (no searches) and stay length-agnostic.
    auto build = [&](std::vector<level_seg_t>& segs,
                     const std::vector<int64_t>& work,
                     i_t col_cap) {
      auto chain_here = [&](i_t l) { return chain_ok[l] && chain_run[l] >= chain_min_levels; };
      auto ok         = [&](i_t l) {
        return width[l] <= max_bundle_cols && work[l] <= max_bundle_work &&
               (col_cap <= 0 || max_col[l] <= col_cap);
      };
      i_t l = 0;
      while (l < n_head_levels_) {
        if (chain_here(l)) {
          segs.push_back({l, l + chain_run[l], 2, ldlt_detail::bundle_block_dim});
          l += chain_run[l];
          continue;
        }
        if (!ok(l)) {
          segs.push_back({l, l + 1, 0, 0});
          l++;
          continue;
        }
        i_t hi           = l;
        int64_t seg_work = 0;
        while (hi < n_head_levels_ && ok(hi) && seg_work + work[hi] <= max_seg_work &&
               !chain_here(hi)) {
          seg_work += work[hi];
          hi++;
        }
        segs.push_back({l, hi, 1, ldlt_detail::bundle_block_dim});
        l = hi;
      }
    };
    build(factor_segs_, factor_work, 0);
    build(fwd_segs_, fwd_work, 0);
    build(bwd_segs_, bwd_work, 0);

    // Chain window metadata: window boundaries (shared-memory capacity
    // driven), per-column data offsets inside the window buffer, the window
    // slot of every row-pattern / column entry (-1 = external), and running
    // in-window op prefixes. The pattern is static, so everything the
    // sequential sweeps need is resolved here once.
    chain_win_.clear();
    chain_smoff_.assign(tail_start_ + 1, 0);
    chain_fop_off_.assign(tail_start_ + 1, 0);
    chain_bop_off_.assign(tail_start_ + 1, 0);
    chain_rp_slot_.assign(nnz_L_, -1);
    chain_col_slot_.assign(nnz_L_, -1);
    for (auto& seg : factor_segs_) {
      if (seg.bundled != 2) { continue; }
      const i_t base    = head_level_ptr_[seg.lo];
      const i_t n_chain = seg.hi - seg.lo;
      seg.win_begin     = static_cast<i_t>(chain_win_.size());
      chain_win_.push_back(0);
      i_t w_start = 0, cum_len = 0, cum_rp = 0;
      for (i_t i = 0; i < n_chain; i++) {
        const i_t j    = head_level_cols_[base + i];
        const i_t clen = Lp_[j + 1] - Lp_[j];
        const i_t rlen = rp_ptr_[j + 1] - rp_ptr_[j];
        if (i > w_start && (cum_len + clen > ldlt_detail::chain_smem ||
                            cum_rp + rlen > ldlt_detail::chain_smem ||
                            i - w_start >= ldlt_detail::chain_window)) {
          chain_win_.push_back(i);
          w_start = i;
          cum_len = 0;
          cum_rp  = 0;
        }
        chain_smoff_[base + i] = cum_len;
        cum_len += clen;
        cum_rp += rlen;
      }
      chain_win_.push_back(n_chain);
      seg.win_count = static_cast<i_t>(chain_win_.size()) - seg.win_begin - 1;

      i_t fop = 0, bop = 0;
      for (i_t w = 0; w < seg.win_count; w++) {
        const i_t w0     = chain_win_[seg.win_begin + w];
        const i_t w1     = chain_win_[seg.win_begin + w + 1];
        const i_t* wcols = head_level_cols_.data() + base + w0;  // ascending along the chain
        const i_t wn     = w1 - w0;
        for (i_t i = w0; i < w1; i++) {
          const i_t j              = head_level_cols_[base + i];
          chain_fop_off_[base + i] = fop;
          for (i_t t = rp_ptr_[j]; t < rp_ptr_[j + 1]; t++) {
            const i_t* it = std::lower_bound(wcols, wcols + wn, rp_col_[t]);
            if (it != wcols + wn && *it == rp_col_[t]) {
              chain_rp_slot_[t] = static_cast<i_t>(it - wcols);
              fop++;
            }
          }
          chain_bop_off_[base + i] = bop;
          for (i_t p = Lp_[j]; p < Lp_[j + 1]; p++) {
            const i_t* it = std::lower_bound(wcols, wcols + wn, Li_[p]);
            if (it != wcols + wn && *it == Li_[p]) {
              chain_col_slot_[p] = static_cast<i_t>(it - wcols);
              bop++;
            }
          }
        }
      }
      chain_fop_off_[base + n_chain] = fop;
      chain_bop_off_[base + n_chain] = bop;
    }
    // The solve plans classify chains identically; share the window metadata.
    for (auto* segs : {&fwd_segs_, &bwd_segs_}) {
      for (auto& seg : *segs) {
        if (seg.bundled != 2) { continue; }
        for (const auto& fs : factor_segs_) {
          if (fs.bundled == 2 && fs.lo == seg.lo && fs.hi == seg.hi) {
            seg.win_begin = fs.win_begin;
            seg.win_count = fs.win_count;
            break;
          }
        }
      }
    }

    if (std::getenv("CUOPT_LDLT_PIVDBG") != nullptr) {
      auto stat = [&](const char* name, const std::vector<level_seg_t>& segs) {
        i_t per = 0, bun = 0, chn = 0;
        for (const auto& s : segs) {
          per += (s.bundled == 0);
          bun += (s.bundled == 1);
          chn += (s.bundled == 2);
        }
        fprintf(stderr,
                "[ldlt] %s segs: %d per-level, %d bundles, %d chains (levels=%d)\n",
                name,
                per,
                bun,
                chn,
                n_head_levels_);
      };
      stat("factor", factor_segs_);
      stat("fwd", fwd_segs_);
      stat("bwd", bwd_segs_);
    }
  }

  // Threshold below which a pivot is replaced rather than divided by:
  // numerically rank-deficient systems (e.g. A*Dinv*A^T with redundant rows)
  // produce (near-)zero pivots that pivoting factorizations absorb; we
  // perturb instead and let iterative refinement clean up the solution.
  static f_t compute_static_pivot_tol(f_t max_abs_diag)
  {
    constexpr f_t rel_tol       = 1e-14;
    constexpr f_t abs_tol_floor = 1e-30;
    return std::max(rel_tol * max_abs_diag, abs_tol_floor);
  }

  // Banded-matrix nested dissection: for a matrix that is banded (half
  // bandwidth bw) in its natural order, AMD keeps a near-natural order whose
  // elimination tree is one n-long chain — the level-scheduled factorization
  // and solves then serialize completely (LISWET-class problems). Recursive
  // index bisection with the bw coupling columns as separators (ordered
  // last) is an exact nested dissection for banded graphs: the etree depth
  // drops to O(bw log n) with geometrically wide levels, at a modest fill
  // increase. Returns false if the matrix is not narrow-banded.
  // Reverse Cuthill-McKee over the symmetric pattern, restricted to the
  // vertices with mask == 0 (spike columns are excluded). Returns the RCM
  // order (vertex list); long thin graphs (staircase/period-coupled
  // dynamics) become narrow-banded in this order even when the natural index
  // order is not.
  std::vector<i_t> compute_rcm_order(const std::vector<char>& mask) const
  {
    std::vector<i_t> order;
    order.reserve(n_);
    std::vector<i_t> deg(n_, 0);
    for (i_t j = 0; j < n_; j++) {
      if (mask[j]) { continue; }
      for (i_t p = a_rowptr_host_[j]; p < a_rowptr_host_[j + 1]; p++) {
        const i_t c = a_colidx_host_[p];
        if (c != j && !mask[c]) { deg[j]++; }
      }
    }
    std::vector<char> visited(mask.begin(), mask.end());
    std::vector<i_t> queue;
    queue.reserve(n_);
    for (i_t start0 = 0; start0 < n_; start0++) {
      if (visited[start0]) { continue; }
      // pseudo-peripheral start: two BFS sweeps from the component's first
      // vertex, restart from the last-visited minimum-degree vertex.
      i_t start = start0;
      for (int sweep = 0; sweep < 2; sweep++) {
        queue.clear();
        queue.push_back(start);
        std::vector<i_t> seen{start};
        std::vector<char> in_bfs(n_, 0);
        in_bfs[start] = 1;
        size_t qh     = 0;
        i_t last      = start;
        while (qh < queue.size()) {
          const i_t v = queue[qh++];
          last        = v;
          for (i_t p = a_rowptr_host_[v]; p < a_rowptr_host_[v + 1]; p++) {
            const i_t c = a_colidx_host_[p];
            if (c == v || visited[c] || in_bfs[c]) { continue; }
            in_bfs[c] = 1;
            queue.push_back(c);
          }
        }
        start = last;
      }
      // RCM BFS from the chosen start, neighbors by ascending degree.
      queue.clear();
      queue.push_back(start);
      visited[start] = 1;
      size_t qh      = 0;
      std::vector<i_t> nbrs;
      while (qh < queue.size()) {
        const i_t v = queue[qh++];
        order.push_back(v);
        nbrs.clear();
        for (i_t p = a_rowptr_host_[v]; p < a_rowptr_host_[v + 1]; p++) {
          const i_t c = a_colidx_host_[p];
          if (c == v || visited[c]) { continue; }
          visited[c] = 1;
          nbrs.push_back(c);
        }
        std::sort(nbrs.begin(), nbrs.end(), [&](i_t a, i_t b) {
          return deg[a] < deg[b] || (deg[a] == deg[b] && a < b);
        });
        for (i_t c : nbrs) {
          queue.push_back(c);
        }
      }
    }
    std::reverse(order.begin(), order.end());
    return order;
  }

  bool compute_banded_nd_ordering()
  {
    constexpr i_t nd_bw_cap   = 40;
    constexpr i_t nd_min_size = 1024;
    if (n_ < nd_min_size) { return false; }
    // Columns owning entries beyond the band cap become "spikes" (e.g. a
    // wrap-around constraint, dense-ish rows): they are ordered last, like a
    // root separator, and the band test applies to the rest.
    std::vector<char> spike(n_, 0);
    for (i_t r = 0; r < n_; r++) {
      for (i_t p = a_rowptr_host_[r]; p < a_rowptr_host_[r + 1]; p++) {
        if (std::abs(r - a_colidx_host_[p]) > nd_bw_cap) {
          spike[r]                 = 1;
          spike[a_colidx_host_[p]] = 1;
        }
      }
    }
    i_t n_spikes = 0;
    for (i_t j = 0; j < n_; j++) {
      n_spikes += spike[j];
    }
    const i_t spike_cap = std::max<i_t>(32, n_ / 64);
    std::vector<i_t> nsq;
    if (n_spikes > spike_cap) {
      // Not banded in the natural order: try the RCM order instead (without
      // spikes — RCM itself decides nothing about them, so retry with a
      // dense-degree-based spike set: vertices whose degree is far above the
      // median act like dense rows).
      std::vector<i_t> sdeg(n_, 0);
      for (i_t r = 0; r < n_; r++) {
        sdeg[r] = a_rowptr_host_[r + 1] - a_rowptr_host_[r];
      }
      std::vector<i_t> tmp(sdeg);
      std::nth_element(tmp.begin(), tmp.begin() + n_ / 2, tmp.end());
      const i_t med_deg = std::max<i_t>(2, tmp[n_ / 2]);
      std::fill(spike.begin(), spike.end(), 0);
      n_spikes = 0;
      for (i_t j = 0; j < n_; j++) {
        if (sdeg[j] > 8 * med_deg) {
          spike[j] = 1;
          n_spikes++;
        }
      }
      if (n_spikes > spike_cap) { return false; }
      nsq = compute_rcm_order(spike);
      // Bandwidth in RCM positions.
      std::vector<i_t> pos_r(n_, -1);
      for (i_t q = 0; q < static_cast<i_t>(nsq.size()); q++) {
        pos_r[nsq[q]] = q;
      }
      i_t bw_rcm = 1;
      for (i_t r = 0; r < n_ && bw_rcm <= nd_bw_cap; r++) {
        if (spike[r]) { continue; }
        for (i_t p = a_rowptr_host_[r]; p < a_rowptr_host_[r + 1]; p++) {
          const i_t c = a_colidx_host_[p];
          if (spike[c]) { continue; }
          bw_rcm = std::max(bw_rcm, std::abs(pos_r[r] - pos_r[c]));
        }
      }
      if (bw_rcm > nd_bw_cap) {
        if (std::getenv("CUOPT_LDLT_PIVDBG") != nullptr) {
          fprintf(stderr,
                  "[ldlt] banded ND rejected: natural spikes=%d, RCM bw=%d (cap %d)\n",
                  n_spikes,
                  bw_rcm,
                  nd_bw_cap);
        }
        return false;
      }
    } else {
      // Positions of non-spike indices; position distance <= index distance,
      // so a bw-wide position separator is a valid banded separator.
      nsq.reserve(n_ - n_spikes);
      for (i_t j = 0; j < n_; j++) {
        if (!spike[j]) { nsq.push_back(j); }
      }
    }
    const i_t nn = static_cast<i_t>(nsq.size());
    std::vector<i_t> pos(n_, -1);
    for (i_t q = 0; q < nn; q++) {
      pos[nsq[q]] = q;
    }
    i_t bw = 1;
    for (i_t r = 0; r < n_; r++) {
      if (spike[r]) { continue; }
      for (i_t p = a_rowptr_host_[r]; p < a_rowptr_host_[r + 1]; p++) {
        const i_t c = a_colidx_host_[p];
        if (spike[c]) { continue; }
        bw = std::max(bw, std::abs(pos[r] - pos[c]));
      }
    }
    if (std::getenv("CUOPT_LDLT_PIVDBG") != nullptr) {
      fprintf(stderr, "[ldlt] banded ND: bw=%d spikes=%d n=%d\n", bw, n_spikes, n_);
    }

    perm_.clear();
    perm_.reserve(n_);
    const i_t leaf_size = std::max<i_t>(2 * bw, 64);
    // Iterative post-order bisection over positions: emit [lo, hi) interiors
    // first, the separator [mid - bw + 1, mid] last.
    struct span_t {
      i_t lo, hi;
      bool expanded;
    };
    std::vector<span_t> stack{{0, nn, false}};
    while (!stack.empty()) {
      span_t s = stack.back();
      stack.pop_back();
      const i_t len = s.hi - s.lo;
      if (len <= 0) { continue; }
      if (!s.expanded && len <= leaf_size) {
        for (i_t q = s.lo; q < s.hi; q++) {
          perm_.push_back(nsq[q]);
        }
        continue;
      }
      if (s.expanded) {
        // children done: emit the separator
        const i_t mid = s.lo + len / 2;
        for (i_t q = mid - bw + 1; q <= mid; q++) {
          perm_.push_back(nsq[q]);
        }
        continue;
      }
      const i_t mid = s.lo + len / 2;
      stack.push_back({s.lo, s.hi, true});           // separator after children
      stack.push_back({mid + 1, s.hi, false});       // right interior
      stack.push_back({s.lo, mid - bw + 1, false});  // left interior
    }
    for (i_t j = 0; j < n_; j++) {
      if (spike[j]) { perm_.push_back(j); }
    }
    cuopt_assert(static_cast<i_t>(perm_.size()) == n_, "banded ND permutation incomplete");
    return static_cast<i_t>(perm_.size()) == n_;
  }

  // Fill-reducing ordering. perm_[k] = original index of the k-th pivot.
  // Narrow-banded matrices use the banded nested dissection above (parallel
  // schedule); everything else uses SuiteSparse AMD (approximate minimum
  // degree) on the full symmetric pattern. The ordering only affects fill-in,
  // never correctness, so any failure falls back to the natural ordering.
  void compute_ordering()
  {
    perm_.resize(n_);
    perm_inv_.resize(n_);
    for (i_t k = 0; k < n_; k++) {
      perm_[k] = k;
    }

    static_assert(std::is_same_v<i_t, int>, "amd_order requires int32 indices");
    if (n_ > 1 && !compute_banded_nd_ordering()) {
      perm_.assign(n_, 0);
      for (i_t k = 0; k < n_; k++) {
        perm_[k] = k;
      }
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

    // Level schedule on the elimination tree (cheap, O(n)); needed up front
    // so the dense tail can be chosen before the symbolic factorization, and
    // the (potentially huge) tail-tail patterns are then never formed.
    std::vector<i_t> level(n_, 0);
    {
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

    // Exact factor column counts (Gilbert-Ng-Peyton) without forming the
    // patterns; they drive the density-based dense-tail choice.
    {
      std::vector<i_t> post = ldlt_detail::ldlt_etree_postorder<i_t>(etree_parent_);
      std::vector<std::vector<i_t>> lc(n_);  // strict lower pattern per column
      for (i_t k = 0; k < n_; k++) {
        for (i_t i : uc[k]) {
          lc[i].push_back(k);
        }
      }
      col_counts_ = ldlt_detail::ldlt_column_counts<i_t>(lc, etree_parent_, post);
    }
    if (halted()) { return CONCURRENT_HALT_RETURN; }
    choose_tail_density();

    // Row patterns of L: rows[j] = sorted { k < j : L(j, k) != 0 }, computed
    // via the elimination-tree reach of the entries of row j (Davis, Direct
    // Methods for Sparse Linear Systems). For tail rows (j >= tail_start_)
    // the reach is clamped at the tail boundary: tail-tail entries live in
    // the dense block and their (potentially huge) sparse patterns are never
    // formed.
    std::vector<std::vector<i_t>> rows(n_);
    std::vector<i_t> col_count(n_, 0);
    {
      std::vector<i_t> mark(n_, -1);
      for (i_t j = 0; j < n_; j++) {
        mark[j] = j;
        for (i_t i : uc[j]) {
          for (i_t node = i; node != -1 && mark[node] != j && node < j;
               node = etree_parent_[node]) {
            if (j >= tail_start_ && node >= tail_start_) { break; }
            rows[j].push_back(node);
            col_count[node]++;
            mark[node] = j;
          }
        }
        std::sort(rows[j].begin(), rows[j].end());
        if ((j & 4095) == 0 && halted()) { return CONCURRENT_HALT_RETURN; }
      }
    }
    // Cross-check the pattern pass against the exact counts: head columns
    // must match (tail columns hold their entries in the dense block).
    if (std::getenv("CUOPT_LDLT_PIVDBG") != nullptr) {
      i_t mismatches = 0;
      for (i_t j = 0; j < tail_start_; j++) {
        if (col_count[j] != col_counts_[j] - 1) { mismatches++; }
      }
      fprintf(stderr,
              "[ldlt] GNP count check: %d mismatches over %d head cols (tail_dim=%d)\n",
              mismatches,
              tail_start_,
              tail_dim_);
    }

    // Column pointers of L (strict lower triangle, CSC).
    int64_t nnz_l64 = 0;
    for (i_t j = 0; j < n_; j++) {
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

    // Level schedule restricted to the head columns [0, tail_start_): a head
    // column's row pattern only references columns below it, so the head
    // factors independently of the tail.
    {
      i_t max_level = 0;
      for (i_t j = 0; j < tail_start_; j++) {
        max_level = std::max(max_level, level[j]);
      }
      n_head_levels_ = tail_start_ > 0 ? max_level + 1 : 0;
      head_level_ptr_.assign(n_head_levels_ + 1, 0);
      for (i_t j = 0; j < tail_start_; j++) {
        head_level_ptr_[level[j] + 1]++;
      }
      for (i_t l = 0; l < n_head_levels_; l++) {
        head_level_ptr_[l + 1] += head_level_ptr_[l];
      }
      head_level_cols_.resize(tail_start_);
      std::vector<i_t> fill_ptr(head_level_ptr_.begin(), head_level_ptr_.end() - 1);
      for (i_t j = 0; j < tail_start_; j++) {
        head_level_cols_[fill_ptr[level[j]]++] = j;
      }
      head_level_max_rp_.assign(n_head_levels_, 0);
      for (i_t j = 0; j < tail_start_; j++) {
        head_level_max_rp_[level[j]] =
          std::max(head_level_max_rp_[level[j]], rp_ptr_[j + 1] - rp_ptr_[j]);
      }
    }

    build_level_bundles();

    // Tail bookkeeping: per head column, the first tail row inside its
    // (sorted) column pattern; per tail row, where the head part of its row
    // pattern ends; and the light/heavy split of the Schur contributions.
    tail_seg_.assign(tail_start_, 0);
    schur_light_.clear();
    schur_heavy_.clear();
    if (tail_dim_ > 0) {
      for (i_t k = 0; k < tail_start_; k++) {
        auto beg     = Li_.begin() + Lp_[k];
        auto end     = Li_.begin() + Lp_[k + 1];
        tail_seg_[k] = static_cast<i_t>(std::lower_bound(beg, end, tail_start_) - Li_.begin());
        i_t t_k      = Lp_[k + 1] - tail_seg_[k];
        if (t_k > 0) {
          if (t_k >= 256) {
            schur_heavy_.push_back(k);
          } else {
            schur_light_.push_back(k);
          }
        }
      }
      rp_head_end_.assign(n_, 0);
      for (i_t j = 0; j < n_; j++) {
        if (j < tail_start_) {
          rp_head_end_[j] = rp_ptr_[j + 1];
        } else {
          auto beg = rp_col_.begin() + rp_ptr_[j];
          auto end = rp_col_.begin() + rp_ptr_[j + 1];
          rp_head_end_[j] =
            static_cast<i_t>(std::lower_bound(beg, end, tail_start_) - rp_col_.begin());
        }
      }
    }
    n_schur_light_ = static_cast<i_t>(schur_light_.size());
    n_schur_heavy_ = static_cast<i_t>(schur_heavy_.size());

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
          a2l_[p] = -(static_cast<int64_t>(pi) + 2);
          continue;
        }
        if (pi < pk) { std::swap(pi, pk); }  // entry (row pi, col pk) of L, pi > pk
        if (pk >= tail_start_) {
          // both endpoints in the dense tail
          a2l_[p] = static_cast<int64_t>(nnz_L_) +
                    static_cast<int64_t>(pk - tail_start_) * tail_dim_ + (pi - tail_start_);
          continue;
        }
        i_t lo  = Lp_[pk];
        i_t hi  = Lp_[pk + 1];
        auto it = std::lower_bound(Li_.begin() + lo, Li_.begin() + hi, pi);
        if (it == Li_.begin() + hi || *it != pi) {
          settings_.log.printf(
            "Internal error: A entry (%d, %d) missing from factor pattern\n", r, c);
          return -1;
        }
        a2l_[p] = static_cast<int64_t>(it - Li_.begin());
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
      d_diag0_.resize(n_, stream);
      d_scale_.resize(n_, stream);
      d_solve_b_.resize(n_, stream);
      d_solve_x_.resize(n_, stream);
      if (solve_graph_valid_) {
        RAFT_CUDA_TRY(cudaGraphExecDestroy(solve_graph_));
        solve_graph_ = nullptr;
      }
      solve_graph_valid_ = false;
      solve_graph_warm_  = false;
      if (factor_graph_valid_) {
        RAFT_CUDA_TRY(cudaGraphExecDestroy(factor_graph_));
        factor_graph_ = nullptr;
      }
      factor_graph_valid_ = false;
      factor_graph_warm_  = false;
      raft::copy(d_Lp_.data(), Lp_.data(), Lp_.size(), stream);
      raft::copy(d_Li_.data(), Li_.data(), Li_.size(), stream);
      raft::copy(d_rp_ptr_.data(), rp_ptr_.data(), rp_ptr_.size(), stream);
      raft::copy(d_rp_col_.data(), rp_col_.data(), rp_col_.size(), stream);
      raft::copy(d_rp_pos_.data(), rp_pos_.data(), rp_pos_.size(), stream);
      raft::copy(d_a2l_.data(), a2l_.data(), a2l_.size(), stream);
      raft::copy(d_level_cols_.data(), level_cols_.data(), level_cols_.size(), stream);
      raft::copy(d_perm_.data(), perm_.data(), perm_.size(), stream);
      if (tail_start_ > 0) {
        d_head_level_cols_.resize(tail_start_, stream);
        raft::copy(
          d_head_level_cols_.data(), head_level_cols_.data(), head_level_cols_.size(), stream);
        d_head_level_ptr_.resize(n_head_levels_ + 1, stream);
        raft::copy(d_head_level_ptr_.data(), head_level_ptr_.data(), n_head_levels_ + 1, stream);
      }
      if (!chain_win_.empty()) {
        d_chain_win_.resize(chain_win_.size(), stream);
        raft::copy(d_chain_win_.data(), chain_win_.data(), chain_win_.size(), stream);
        d_chain_smoff_.resize(chain_smoff_.size(), stream);
        raft::copy(d_chain_smoff_.data(), chain_smoff_.data(), chain_smoff_.size(), stream);
        d_chain_fop_off_.resize(chain_fop_off_.size(), stream);
        raft::copy(d_chain_fop_off_.data(), chain_fop_off_.data(), chain_fop_off_.size(), stream);
        d_chain_bop_off_.resize(chain_bop_off_.size(), stream);
        raft::copy(d_chain_bop_off_.data(), chain_bop_off_.data(), chain_bop_off_.size(), stream);
        d_chain_rp_slot_.resize(chain_rp_slot_.size(), stream);
        raft::copy(d_chain_rp_slot_.data(), chain_rp_slot_.data(), chain_rp_slot_.size(), stream);
        d_chain_col_slot_.resize(chain_col_slot_.size(), stream);
        raft::copy(
          d_chain_col_slot_.data(), chain_col_slot_.data(), chain_col_slot_.size(), stream);
      }
      if (tail_dim_ > 0) {
        d_dense_.resize(static_cast<size_t>(tail_dim_) * tail_dim_, stream);
        d_tail_seg_.resize(std::max<i_t>(tail_start_, 1), stream);
        if (tail_start_ > 0) {
          raft::copy(d_tail_seg_.data(), tail_seg_.data(), tail_seg_.size(), stream);
        }
        d_rp_head_end_.resize(n_, stream);
        raft::copy(d_rp_head_end_.data(), rp_head_end_.data(), rp_head_end_.size(), stream);
        if (n_schur_light_ > 0) {
          d_schur_light_.resize(n_schur_light_, stream);
          raft::copy(d_schur_light_.data(), schur_light_.data(), n_schur_light_, stream);
        }
        if (n_schur_heavy_ > 0) {
          d_schur_heavy_.resize(n_schur_heavy_, stream);
          raft::copy(d_schur_heavy_.data(), schur_heavy_.data(), n_schur_heavy_, stream);
        }
        // Workspace for the heavy-column Schur GEMM and the panel update:
        // at least one factor panel (nb columns), at most ~512 MB for V + W.
        const double ws_budget = 512.0 * 1024.0 * 1024.0;
        schur_chunk_           = std::max<i_t>(
          ldlt_detail::dense_nb,
          std::min<i_t>(n_schur_heavy_ > 0 ? n_schur_heavy_ : ldlt_detail::dense_nb,
                        static_cast<i_t>(ws_budget / (2.0 * sizeof(f_t) * tail_dim_))));
        d_panelV_.resize(static_cast<size_t>(tail_dim_) * schur_chunk_, stream);
        d_panelW_.resize(static_cast<size_t>(tail_dim_) * schur_chunk_, stream);
      }
      RAFT_CUDA_TRY(cudaStreamSynchronize(stream));
    }

    f_t symbolic_time = toc(start_symbolic);
    settings_.log.printf("Symbolic factorization time : %.2fs\n", symbolic_time);
    settings_.log.printf("Symbolic nonzeros in factor : %.2e\n",
                         static_cast<f_t>(nnz_L_) + static_cast<f_t>(n_));
    settings_.log.printf("Elimination tree levels     : %d\n", n_levels_);

    if (std::getenv("CUOPT_LDLT_STATS") != nullptr) { log_structure_stats(); }

    analyzed_ = true;
    return 0;
  }

  // Logs how the factor's nonzeros distribute over trailing column blocks
  // and over the level schedule (CUOPT_LDLT_STATS=1). Drives the dense-tail
  // optimization decisions.
  void log_structure_stats() const
  {
    auto& log = settings_.log;
    log.printf("LDLT stats: n=%d nnz_L=%d levels=%d tail_dim=%d head_levels=%d light=%d heavy=%d\n",
               n_,
               nnz_L_,
               n_levels_,
               tail_dim_,
               n_head_levels_,
               n_schur_light_,
               n_schur_heavy_);
    // Trailing-block density: how dense are the last d columns of L?
    for (double frac : {0.005, 0.01, 0.02, 0.05, 0.10, 0.20}) {
      i_t d = std::max<i_t>(1, static_cast<i_t>(frac * n_));
      i_t j0 = n_ - d;
      int64_t nnz_block = 0;  // entries (i, j) with j >= j0 (then i >= j > j0 too)
      int64_t nnz_cols  = 0;  // all entries in columns >= j0
      for (i_t j = j0; j < n_; j++) {
        nnz_cols += Lp_[j + 1] - Lp_[j];
        nnz_block += Lp_[j + 1] - Lp_[j];  // all rows of col j>=j0 are > j >= j0
      }
      // entries of leading columns whose row lands in the tail
      int64_t nnz_into_tail = 0;
      for (i_t j = 0; j < j0; j++) {
        // count rows >= j0 in column j (binary search)
        auto beg = Li_.begin() + Lp_[j];
        auto end = Li_.begin() + Lp_[j + 1];
        nnz_into_tail += end - std::lower_bound(beg, end, j0);
      }
      double dens = static_cast<double>(nnz_block) / (0.5 * double(d) * double(d + 1));
      log.printf(
        "LDLT stats: tail d=%7d (%4.1f%%)  nnz(tail cols)=%10lld  density=%6.3f  "
        "nnz(rows into tail from head)=%10lld  tail share of L=%5.1f%%\n",
        d,
        100.0 * frac,
        static_cast<long long>(nnz_cols),
        dens,
        static_cast<long long>(nnz_into_tail),
        100.0 * double(nnz_cols) / double(std::max<int64_t>(1, nnz_L_)));
    }
    // Level profile: how much work sits in the last (deepest) levels?
    int64_t total_work = 0;
    std::vector<int64_t> level_work(n_levels_, 0);
    for (i_t l = 0; l < n_levels_; l++) {
      for (i_t t = level_ptr_[l]; t < level_ptr_[l + 1]; t++) {
        i_t j = level_cols_[t];
        // work ~ sum over contributing columns of their below-j lengths;
        // approximate by the row-pattern length times average column length
        level_work[l] += rp_ptr_[j + 1] - rp_ptr_[j];
      }
      total_work += level_work[l];
    }
    i_t singleton_levels = 0;
    for (i_t l = 0; l < n_levels_; l++) {
      if (level_ptr_[l + 1] - level_ptr_[l] <= 2) { singleton_levels++; }
    }
    log.printf("LDLT stats: levels with <=2 cols: %d / %d\n", singleton_levels, n_levels_);
    // cumulative share of row-pattern entries in the deepest 1%/5%/20% of levels
    for (double frac : {0.01, 0.05, 0.20}) {
      i_t l0       = n_levels_ - std::max<i_t>(1, static_cast<i_t>(frac * n_levels_));
      int64_t work = 0;
      i_t cols     = 0;
      for (i_t l = l0; l < n_levels_; l++) {
        work += level_work[l];
        cols += level_ptr_[l + 1] - level_ptr_[l];
      }
      log.printf("LDLT stats: deepest %4.1f%% levels: cols=%8d  rowpat share=%5.1f%%\n",
                 100.0 * frac,
                 cols,
                 100.0 * double(work) / double(std::max<int64_t>(1, total_work)));
    }
  }

  // Numeric factorization on the device with level-scheduled kernels.
  // a_values is a device pointer holding the nnz_A_ values of the analyzed
  // matrix in its original entry order.
  // Issues the full numeric-factorization kernel sequence (no host reads):
  // used both eagerly and under CUDA-graph capture. static_pivot_tol must be
  // supplied (the Jacobi-scaled path has max|diag| == 1 by construction).
  // halt checks only run when allow_halt (capture must not early-return).
  i_t factor_kernels_device(const f_t* a_values,
                            f_t static_pivot_tol_in,
                            bool allow_halt,
                            rmm::cuda_stream_view stream)
  {
    RAFT_CUDA_TRY(cudaMemsetAsync(d_Lx_.data(), 0, sizeof(f_t) * nnz_L_, stream));
    RAFT_CUDA_TRY(cudaMemsetAsync(d_D_.data(), 0, sizeof(f_t) * n_, stream));
    if (tail_dim_ > 0) {
      RAFT_CUDA_TRY(cudaMemsetAsync(
        d_dense_.data(), 0, sizeof(f_t) * static_cast<size_t>(tail_dim_) * tail_dim_, stream));
    }
    d_fail_.set_value_to_zero_async(stream);
    d_static_pivots_.set_value_to_zero_async(stream);
    d_sign_corrections_.set_value_to_zero_async(stream);
    last_sign_corrections_ = 0;

    constexpr int block_dim = ldlt_detail::factor_block_dim;
    if (nnz_A_ > 0) {
      const int grid = (nnz_A_ + block_dim - 1) / block_dim;
      ldlt_detail::ldlt_scatter_kernel<i_t, f_t><<<grid, block_dim, 0, stream>>>(
        a_values,
        d_a2l_.data(),
        nnz_A_,
        nnz_L_,
        d_Lx_.data(),
        tail_dim_ > 0 ? d_dense_.data() : d_Lx_.data(),  // unused when no tail
        d_D_.data());
      RAFT_CHECK_CUDA(stream);
    }
    // Snapshot of the assembled (pre-update) diagonal: per-column scale for
    // the SPD drop test.
    RAFT_CUDA_TRY(cudaMemcpyAsync(
      d_diag0_.data(), d_D_.data(), sizeof(f_t) * n_, cudaMemcpyDeviceToDevice, stream));

    // Symmetric Jacobi scaling of the quasi-definite KKT system (M' = S M S):
    // bounds the diagonal to +-1, taming element growth of the pivot-free
    // factorization on badly ranged systems; the solve untransforms.
    if (scaling_active_) {
      const int grid_n0 = (n_ + block_dim - 1) / block_dim;
      ldlt_detail::ldlt_compute_scale_kernel<i_t, f_t>
        <<<grid_n0, block_dim, 0, stream>>>(d_D_.data(), n_, d_scale_.data());
      RAFT_CHECK_CUDA(stream);
      ldlt_detail::ldlt_apply_scale_kernel<i_t, f_t><<<n_, block_dim, 0, stream>>>(
        d_Lp_.data(), d_Li_.data(), n_, d_scale_.data(), d_Lx_.data(), d_D_.data());
      RAFT_CHECK_CUDA(stream);
      if (tail_dim_ > 0) {
        const int64_t elems = static_cast<int64_t>(tail_dim_) * tail_dim_;
        const int grid      = static_cast<int>((elems + block_dim - 1) / block_dim);
        ldlt_detail::ldlt_apply_scale_dense_kernel<i_t, f_t><<<grid, block_dim, 0, stream>>>(
          tail_start_, tail_dim_, d_scale_.data(), d_dense_.data());
        RAFT_CHECK_CUDA(stream);
      }
    }

    // Static pivoting threshold: supplied by the caller (graph capture; the
    // scaled system has max|diag| == 1), or computed from the assembled
    // diagonal (host read - eager paths only).
    const f_t static_pivot_tol =
      static_pivot_tol_in > f_t(0)
        ? static_pivot_tol_in
        : compute_static_pivot_tol(thrust::transform_reduce(rmm::exec_policy(stream),
                                                            d_D_.data(),
                                                            d_D_.data() + n_,
                                                            ldlt_detail::abs_op<f_t>{},
                                                            f_t(0),
                                                            thrust::maximum<f_t>{}));

    // The two-phase atomic path parallelizes the updates of long columns over
    // the whole device; it is not bitwise deterministic, so the deterministic
    // mode keeps the sequential-per-column kernel. Light consecutive levels
    // are fused into single-launch bundles (deterministic by construction).
    const bool atomic_head = !settings_.cudss_deterministic;
    for (size_t seg_idx = 0; seg_idx < factor_segs_.size(); seg_idx++) {
      const level_seg_t& seg = factor_segs_[seg_idx];
      if (seg.bundled == 2) {
        const i_t base = head_level_ptr_[seg.lo];
        ldlt_detail::ldlt_factor_chain_kernel<i_t, f_t>
          <<<1, seg.block_dim, 0, stream>>>(d_head_level_cols_.data() + base,
                                            d_chain_win_.data() + seg.win_begin,
                                            seg.win_count,
                                            d_chain_smoff_.data() + base,
                                            d_chain_fop_off_.data() + base,
                                            d_chain_rp_slot_.data(),
                                            d_Lp_.data(),
                                            d_Li_.data(),
                                            d_rp_ptr_.data(),
                                            d_rp_col_.data(),
                                            d_rp_pos_.data(),
                                            d_Lx_.data(),
                                            d_D_.data(),
                                            d_fail_.data(),
                                            d_static_pivots_.data(),
                                            static_pivot_tol,
                                            positive_definite_,
                                            d_perm_.data(),
                                            pivot_n_neg_,
                                            pivot_neg_floor_,
                                            pivot_pos_floor_,
                                            d_diag0_.data(),
                                            scaling_active_ ? d_scale_.data() : nullptr,
                                            d_sign_corrections_.data());
        RAFT_CHECK_CUDA(stream);
        if (allow_halt && (seg_idx & 63) == 0 && halted()) { return CONCURRENT_HALT_RETURN; }
        continue;
      }
      if (seg.bundled == 1) {
        ldlt_detail::ldlt_factor_bundle_kernel<i_t, f_t>
          <<<1, seg.block_dim, 0, stream>>>(d_head_level_ptr_.data(),
                                            d_head_level_cols_.data(),
                                            seg.lo,
                                            seg.hi,
                                            d_Lp_.data(),
                                            d_Li_.data(),
                                            d_rp_ptr_.data(),
                                            d_rp_col_.data(),
                                            d_rp_pos_.data(),
                                            d_Lx_.data(),
                                            d_D_.data(),
                                            d_fail_.data(),
                                            d_static_pivots_.data(),
                                            static_pivot_tol,
                                            positive_definite_,
                                            d_perm_.data(),
                                            pivot_n_neg_,
                                            pivot_neg_floor_,
                                            pivot_pos_floor_,
                                            d_diag0_.data(),
                                            scaling_active_ ? d_scale_.data() : nullptr,
                                            d_sign_corrections_.data());
        RAFT_CHECK_CUDA(stream);
        if (allow_halt && (seg_idx & 63) == 0 && halted()) { return CONCURRENT_HALT_RETURN; }
        continue;
      }
      for (i_t l = seg.lo; l < seg.hi; l++) {
      const i_t level_start = head_level_ptr_[l];
      const i_t level_size  = head_level_ptr_[l + 1] - level_start;
      if (level_size == 0) { continue; }
      if (atomic_head) {
        const i_t n_slices =
          (head_level_max_rp_[l] + ldlt_detail::factor_k_slice - 1) / ldlt_detail::factor_k_slice;
        if (n_slices > 0) {
          dim3 grid(level_size, n_slices);
          ldlt_detail::ldlt_factor_level_update_kernel<i_t, f_t>
            <<<grid, block_dim, 0, stream>>>(d_head_level_cols_.data(),
                                             level_start,
                                             d_Lp_.data(),
                                             d_Li_.data(),
                                             d_rp_ptr_.data(),
                                             d_rp_col_.data(),
                                             d_rp_pos_.data(),
                                             d_Lx_.data(),
                                             d_D_.data());
          RAFT_CHECK_CUDA(stream);
        }
        ldlt_detail::ldlt_factor_level_finalize_kernel<i_t, f_t>
          <<<level_size, block_dim, 0, stream>>>(d_head_level_cols_.data(),
                                                 level_start,
                                                 d_Lp_.data(),
                                                 d_Lx_.data(),
                                                 d_D_.data(),
                                                 d_fail_.data(),
                                                 d_static_pivots_.data(),
                                                 static_pivot_tol,
                                                 positive_definite_,
                                                 d_perm_.data(),
                                                 pivot_n_neg_,
                                                 pivot_neg_floor_,
                                                 pivot_pos_floor_,
                                                 d_diag0_.data(),
                                                 scaling_active_ ? d_scale_.data() : nullptr,
                                                 d_sign_corrections_.data());
        RAFT_CHECK_CUDA(stream);
      } else {
        ldlt_detail::ldlt_factor_level_kernel<i_t, f_t>
          <<<level_size, block_dim, 0, stream>>>(d_head_level_cols_.data(),
                                                 level_start,
                                                 d_Lp_.data(),
                                                 d_Li_.data(),
                                                 d_rp_ptr_.data(),
                                                 d_rp_col_.data(),
                                                 d_rp_pos_.data(),
                                                 d_Lx_.data(),
                                                 d_D_.data(),
                                                 d_fail_.data(),
                                                 d_static_pivots_.data(),
                                                 static_pivot_tol,
                                                 positive_definite_,
                                                 d_perm_.data(),
                                                 pivot_n_neg_,
                                                 pivot_neg_floor_,
                                                 pivot_pos_floor_,
                                                 d_diag0_.data(),
                                                 scaling_active_ ? d_scale_.data() : nullptr,
                                                 d_sign_corrections_.data());
        RAFT_CHECK_CUDA(stream);
      }
      if (allow_halt && (l & 63) == 0 && halted()) { return CONCURRENT_HALT_RETURN; }
      }
    }

    if (tail_dim_ > 0) {
      i_t status = factor_dense_tail(stream, static_pivot_tol, allow_halt);
      if (status != 0) { return status; }
    }
    return 0;
  }

  i_t factorize_values_device(const f_t* a_values)
  {
    f_t start_numeric = tic();
    factorized_       = false;
    auto stream       = handle_ptr_->get_stream();
    // Scale the quasi-definite and legacy (initial-point augmented) systems;
    // the SPD/ADAT path keeps its exact historical arithmetic.
    scaling_active_ = pivot_n_neg_ != 0;

    // The Jacobi-scaled path has max|diag| == 1 by construction, making the
    // static tolerance (and thus the whole kernel sequence) constant: capture
    // it into a CUDA graph keyed on the pivot floors and the value pointer.
    i_t status;
    if (scaling_active_ && !use_host_numeric_) {
      const f_t static_pivot_tol = compute_static_pivot_tol(f_t(1));
      const bool key_match = factor_graph_valid_ && factor_graph_values_ == a_values &&
                             factor_graph_neg_floor_ == pivot_neg_floor_ &&
                             factor_graph_pos_floor_ == pivot_pos_floor_ &&
                             factor_graph_n_neg_ == pivot_n_neg_;
      if (factor_graph_valid_ && !key_match) {
        RAFT_CUDA_TRY(cudaGraphExecDestroy(factor_graph_));
        factor_graph_       = nullptr;
        factor_graph_valid_ = false;
        // keep factor_graph_warm_: library workspaces stay warm
      }
      if (!factor_graph_valid_) {
        if (!factor_graph_warm_) {
          status             = factor_kernels_device(a_values, static_pivot_tol, true, stream);
          factor_graph_warm_ = true;
        } else {
          RAFT_CUDA_TRY(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal));
          status            = factor_kernels_device(a_values, static_pivot_tol, false, stream);
          cudaGraph_t graph = nullptr;
          RAFT_CUDA_TRY(cudaStreamEndCapture(stream, &graph));
          if (status == 0) {
            RAFT_CUDA_TRY(cudaGraphInstantiate(&factor_graph_, graph, nullptr, nullptr, 0));
            factor_graph_valid_     = true;
            factor_graph_values_    = a_values;
            factor_graph_neg_floor_ = pivot_neg_floor_;
            factor_graph_pos_floor_ = pivot_pos_floor_;
            factor_graph_n_neg_     = pivot_n_neg_;
            RAFT_CUDA_TRY(cudaGraphLaunch(factor_graph_, stream));
          }
          RAFT_CUDA_TRY(cudaGraphDestroy(graph));
        }
      } else {
        RAFT_CUDA_TRY(cudaGraphLaunch(factor_graph_, stream));
        status = 0;
      }
    } else {
      // Unscaled paths: the tolerance depends on the assembled diagonal and
      // is computed inside (negative sentinel).
      status = factor_kernels_device(a_values, f_t(-1), true, stream);
    }
    if (status != 0) { return status; }

    i_t fail = d_fail_.value(stream);
    if (halted()) { return CONCURRENT_HALT_RETURN; }
    if (fail != 0) {
      settings_.log.printf("Factorization failed: invalid pivot at column %d\n", fail - 1);
      return -1;
    }
    i_t static_pivots      = d_static_pivots_.value(stream);
    last_sign_corrections_ = d_sign_corrections_.value(stream);
    if (std::getenv("CUOPT_LDLT_PIVDBG") != nullptr) {
      fprintf(stderr,
              "[ldlt] n_neg=%d static=%d signfix=%d scaled=%d negf=%.2e posf=%.2e\n",
              pivot_n_neg_,
              static_pivots,
              last_sign_corrections_,
              scaling_active_ ? 1 : 0,
              pivot_neg_floor_,
              pivot_pos_floor_);
    }
    if (static_pivots > 0) {
      settings_.log.debug("Static pivoting perturbed %d pivots (%d sign corrections)\n",
                          static_pivots,
                          last_sign_corrections_);
    }
    factor_time_ += toc(start_numeric);
    n_factor_calls_++;

    if (first_factor_) {
      settings_.log.debug("Factorization time          : %.2fs\n", toc(start_numeric));
      first_factor_ = false;
    }
    factorized_ = true;
    return 0;
  }

  // Schur assembly plus blocked dense LDL^T of the trailing block. The head
  // factor is complete at this point; light head columns scatter their
  // pair contributions with atomics, heavy ones are gathered into dense
  // panels and applied as DGEMMs. The block itself is then factored with a
  // right-looking panel loop (custom diagonal/panel kernels + DGEMM trailing
  // updates).
  i_t factor_dense_tail(rmm::cuda_stream_view stream, f_t static_pivot_tol, bool allow_halt)
  {
    constexpr int block_dim = ldlt_detail::factor_block_dim;
    const i_t d             = tail_dim_;
    cublasHandle_t cublas   = handle_ptr_->get_cublas_handle();
    RAFT_CUBLAS_TRY(cublasSetStream(cublas, stream));
    cublasPointerMode_t prev_mode;
    RAFT_CUBLAS_TRY(cublasGetPointerMode(cublas, &prev_mode));
    RAFT_CUBLAS_TRY(cublasSetPointerMode(cublas, CUBLAS_POINTER_MODE_HOST));
    const f_t one = 1.0, minus_one = -1.0;

    if (n_schur_light_ > 0) {
      ldlt_detail::ldlt_schur_pairs_kernel<i_t, f_t>
        <<<n_schur_light_, 256, 0, stream>>>(d_schur_light_.data(),
                                             d_Lp_.data(),
                                             d_Li_.data(),
                                             d_tail_seg_.data(),
                                             d_Lx_.data(),
                                             d_D_.data(),
                                             tail_start_,
                                             d,
                                             d_dense_.data(),
                                             d_D_.data());
      RAFT_CHECK_CUDA(stream);
    }

    for (i_t c0 = 0; c0 < n_schur_heavy_; c0 += schur_chunk_) {
      const i_t csz = std::min<i_t>(schur_chunk_, n_schur_heavy_ - c0);
      RAFT_CUDA_TRY(cudaMemsetAsync(
        d_panelV_.data(), 0, sizeof(f_t) * static_cast<size_t>(d) * csz, stream));
      RAFT_CUDA_TRY(cudaMemsetAsync(
        d_panelW_.data(), 0, sizeof(f_t) * static_cast<size_t>(d) * csz, stream));
      ldlt_detail::ldlt_schur_gather_kernel<i_t, f_t>
        <<<csz, block_dim, 0, stream>>>(d_schur_heavy_.data(),
                                        c0,
                                        d_Lp_.data(),
                                        d_Li_.data(),
                                        d_tail_seg_.data(),
                                        d_Lx_.data(),
                                        d_D_.data(),
                                        tail_start_,
                                        d,
                                        d_panelV_.data(),
                                        d_panelW_.data());
      RAFT_CHECK_CUDA(stream);
      RAFT_CUBLAS_TRY(cublasDgemm(cublas,
                                  CUBLAS_OP_N,
                                  CUBLAS_OP_T,
                                  d,
                                  d,
                                  csz,
                                  &minus_one,
                                  d_panelV_.data(),
                                  d,
                                  d_panelW_.data(),
                                  d,
                                  &one,
                                  d_dense_.data(),
                                  d));
      if (allow_halt && halted()) {
        RAFT_CUBLAS_TRY(cublasSetPointerMode(cublas, prev_mode));
        return CONCURRENT_HALT_RETURN;
      }
    }

    // Blocked left-looking dense LDL^T: each panel is first updated with one
    // DGEMM against all previously factored columns (large-k GEMM, half the
    // flops of a full right-looking trailing update), then factored locally.
    constexpr i_t nb = ldlt_detail::dense_nb;
    for (i_t p0 = 0; p0 < d; p0 += nb) {
      const i_t nb_eff = std::min<i_t>(nb, d - p0);
      const i_t m_rows = d - p0;
      if (p0 > 0) {
        // M = (L(p0:p0+nb, 0:p0) * D(0:p0))^T, then
        // S(p0:, p0:p0+nb) -= L(p0:, 0:p0) * M
        {
          const int64_t elems = static_cast<int64_t>(p0) * nb_eff;
          const int grid      = static_cast<int>((elems + block_dim - 1) / block_dim);
          ldlt_detail::ldlt_dense_panel_slab_kernel<i_t, f_t>
            <<<grid, block_dim, 0, stream>>>(
              d_dense_.data(), d, tail_start_, p0, nb_eff, d_D_.data(), d_panelW_.data());
          RAFT_CHECK_CUDA(stream);
        }
        RAFT_CUBLAS_TRY(cublasDgemm(cublas,
                                    CUBLAS_OP_N,
                                    CUBLAS_OP_N,
                                    m_rows,
                                    nb_eff,
                                    p0,
                                    &minus_one,
                                    d_dense_.data() + p0,
                                    d,
                                    d_panelW_.data(),
                                    p0,
                                    &one,
                                    d_dense_.data() + static_cast<int64_t>(p0) * d + p0,
                                    d));
      }
      ldlt_detail::ldlt_dense_diag_kernel<i_t, f_t>
        <<<1, 256, 0, stream>>>(d_dense_.data(),
                                d,
                                tail_start_,
                                p0,
                                nb_eff,
                                d_D_.data(),
                                d_fail_.data(),
                                d_static_pivots_.data(),
                                static_pivot_tol,
                                positive_definite_,
                                d_perm_.data(),
                                pivot_n_neg_,
                                pivot_neg_floor_,
                                pivot_pos_floor_,
                                d_diag0_.data(),
                                scaling_active_ ? d_scale_.data() : nullptr,
                                d_sign_corrections_.data());
      RAFT_CHECK_CUDA(stream);
      const i_t m_rest = d - p0 - nb_eff;
      if (m_rest > 0) {
        const int grid      = (m_rest + block_dim - 1) / block_dim;
        const size_t smem_b = sizeof(f_t) * (static_cast<size_t>(nb_eff) * nb_eff + nb_eff);
        ldlt_detail::ldlt_dense_panel_solve_kernel<i_t, f_t>
          <<<grid, block_dim, smem_b, stream>>>(
            d_dense_.data(), d, tail_start_, p0, nb_eff, d_D_.data());
        RAFT_CHECK_CUDA(stream);
      }
      if (allow_halt && (p0 / nb) % 16 == 0 && halted()) {
        RAFT_CUBLAS_TRY(cublasSetPointerMode(cublas, prev_mode));
        return CONCURRENT_HALT_RETURN;
      }
    }
    RAFT_CUBLAS_TRY(cublasSetPointerMode(cublas, prev_mode));
    return 0;
  }

  // Numeric factorization (host reference implementation).
  // Left-looking by column: for each column j (ascending), apply the updates
  // of all columns k with L(j, k) != 0, then scale by the pivot d_j.
  i_t factorize_values_host()
  {
    f_t start_numeric = tic();
    factorized_       = false;
    scaling_active_   = false;  // the host reference path never scales

    // Scatter A into the factor storage (assignment, see a2l_ comment).
    std::fill(Lx_.begin(), Lx_.end(), f_t(0));
    std::fill(D_.begin(), D_.end(), f_t(0));
    for (i_t e = 0; e < nnz_A_; e++) {
      int64_t m = a2l_[e];
      if (m >= 0) {
        Lx_[m] = a_values_host_[e];  // the host path never uses a dense tail
      } else if (m <= -2) {
        D_[-(m + 2)] = a_values_host_[e];
      }
    }

    f_t max_abs_diag = f_t(0);
    for (i_t j = 0; j < n_; j++) {
      max_abs_diag = std::max(max_abs_diag, std::abs(D_[j]));
    }
    const std::vector<f_t> diag0(D_.begin(), D_.end());
    const f_t static_pivot_tol = compute_static_pivot_tol(max_abs_diag);
    i_t static_pivots          = 0;
    last_sign_corrections_     = 0;

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
      // Pivot (host mirror of ldlt_detail::ldlt_pivot_rule).
      f_t dj = D_[j];
      if (!std::isfinite(dj)) {
        settings_.log.printf("Factorization failed: pivot %e at column %d\n", dj, j);
        return -1;
      }
      if (pivot_n_neg_ <= i_t(-2)) {
        if (positive_definite_ && dj <= f_t(0)) {
          settings_.log.printf("Factorization failed: pivot %e at column %d\n", dj, j);
          return -1;
        }
        if (std::abs(dj) < static_pivot_tol) {
          dj = (dj < f_t(0)) ? -static_pivot_tol : static_pivot_tol;
          static_pivots++;
        }
      } else if (pivot_n_neg_ == i_t(0)) {
        const f_t tol_j = f_t(1e-14) * std::abs(diag0[j]) + f_t(1e-30);
        if (dj >= f_t(0) && dj < tol_j) {
          dj = f_t(ldlt_detail::drop_pivot_value);
          static_pivots++;
        } else if (dj < f_t(0) && -dj < tol_j) {
          last_sign_corrections_++;
          dj = -tol_j;
          static_pivots++;
        }
      } else {
        const bool neg  = perm_[j] < pivot_n_neg_;
        const f_t s     = neg ? f_t(-1) : f_t(1);
        const f_t f_blk = neg ? pivot_neg_floor_ : pivot_pos_floor_;
        const f_t fl    = f_blk > f_t(0) ? f_blk : static_pivot_tol;
        if (s * dj < fl) {
          if (s * dj < f_t(0)) { last_sign_corrections_++; }
          dj = s * fl;
          static_pivots++;
        }
      }
      D_[j] = dj;
      for (i_t p = Lp_[j]; p < Lp_[j + 1]; p++) {
        Lx_[p] /= dj;
      }
      if ((j & 1023) == 0 && halted()) { return CONCURRENT_HALT_RETURN; }
    }
    if (static_pivots > 0) {
      settings_.log.debug("Static pivoting perturbed %d pivots (%d sign corrections)\n",
                          static_pivots,
                          last_sign_corrections_);
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
  // Issues the full solve kernel sequence (no synchronization): used both
  // eagerly and under CUDA-graph capture.
  void solve_kernels_device(const f_t* d_b, f_t* d_x, rmm::cuda_stream_view stream)
  {
    constexpr int block_dim = ldlt_detail::factor_block_dim;
    const int grid_n        = (n_ + block_dim - 1) / block_dim;

    ldlt_detail::ldlt_perm_gather_kernel<i_t, f_t><<<grid_n, block_dim, 0, stream>>>(
      d_b, d_perm_.data(), n_, scaling_active_ ? d_scale_.data() : nullptr, d_work_.data());
    RAFT_CHECK_CUDA(stream);

    cublasHandle_t cublas         = nullptr;
    cublasPointerMode_t prev_mode = CUBLAS_POINTER_MODE_HOST;
    const f_t one                 = 1.0;
    if (tail_dim_ > 0) {
      cublas = handle_ptr_->get_cublas_handle();
      RAFT_CUBLAS_TRY(cublasSetStream(cublas, stream));
      RAFT_CUBLAS_TRY(cublasGetPointerMode(cublas, &prev_mode));
      RAFT_CUBLAS_TRY(cublasSetPointerMode(cublas, CUBLAS_POINTER_MODE_HOST));
    }

    for (const level_seg_t& seg : fwd_segs_) {
      if (seg.bundled == 2) {
        const i_t base = head_level_ptr_[seg.lo];
        ldlt_detail::ldlt_forward_chain_kernel<i_t, f_t>
          <<<1, seg.block_dim, 0, stream>>>(d_head_level_cols_.data() + base,
                                            d_chain_win_.data() + seg.win_begin,
                                            seg.win_count,
                                            d_chain_fop_off_.data() + base,
                                            d_chain_rp_slot_.data(),
                                            d_rp_ptr_.data(),
                                            d_rp_col_.data(),
                                            d_rp_pos_.data(),
                                            d_Lx_.data(),
                                            d_work_.data());
        RAFT_CHECK_CUDA(stream);
        continue;
      }
      if (seg.bundled == 1) {
        ldlt_detail::ldlt_forward_bundle_kernel<i_t, f_t>
          <<<1, seg.block_dim, 0, stream>>>(d_head_level_ptr_.data(),
                                            d_head_level_cols_.data(),
                                            seg.lo,
                                            seg.hi,
                                            d_rp_ptr_.data(),
                                            d_rp_col_.data(),
                                            d_rp_pos_.data(),
                                            d_Lx_.data(),
                                            d_work_.data());
        RAFT_CHECK_CUDA(stream);
        continue;
      }
      for (i_t l = seg.lo; l < seg.hi; l++) {
        const i_t level_start = head_level_ptr_[l];
        const i_t level_size  = head_level_ptr_[l + 1] - level_start;
        if (level_size == 0) { continue; }
        ldlt_detail::ldlt_forward_level_kernel<i_t, f_t>
          <<<level_size, block_dim, 0, stream>>>(d_head_level_cols_.data(),
                                                 level_start,
                                                 d_rp_ptr_.data(),
                                                 d_rp_col_.data(),
                                                 d_rp_pos_.data(),
                                                 d_Lx_.data(),
                                                 d_work_.data());
        RAFT_CHECK_CUDA(stream);
      }
    }

    if (tail_dim_ > 0) {
      if (tail_start_ > 0) {
        ldlt_detail::ldlt_forward_tail_couple_kernel<i_t, f_t>
          <<<tail_dim_, block_dim, 0, stream>>>(tail_start_,
                                                d_rp_ptr_.data(),
                                                d_rp_head_end_.data(),
                                                d_rp_col_.data(),
                                                d_rp_pos_.data(),
                                                d_Lx_.data(),
                                                d_work_.data());
        RAFT_CHECK_CUDA(stream);
      }
      RAFT_CUBLAS_TRY(cublasDtrsm(cublas,
                                  CUBLAS_SIDE_LEFT,
                                  CUBLAS_FILL_MODE_LOWER,
                                  CUBLAS_OP_N,
                                  CUBLAS_DIAG_UNIT,
                                  tail_dim_,
                                  1,
                                  &one,
                                  d_dense_.data(),
                                  tail_dim_,
                                  d_work_.data() + tail_start_,
                                  tail_dim_));
    }

    ldlt_detail::ldlt_diag_scale_kernel<i_t, f_t>
      <<<grid_n, block_dim, 0, stream>>>(d_D_.data(), n_, d_work_.data());
    RAFT_CHECK_CUDA(stream);

    if (tail_dim_ > 0) {
      RAFT_CUBLAS_TRY(cublasDtrsm(cublas,
                                  CUBLAS_SIDE_LEFT,
                                  CUBLAS_FILL_MODE_LOWER,
                                  CUBLAS_OP_T,
                                  CUBLAS_DIAG_UNIT,
                                  tail_dim_,
                                  1,
                                  &one,
                                  d_dense_.data(),
                                  tail_dim_,
                                  d_work_.data() + tail_start_,
                                  tail_dim_));
    }

    for (auto seg_it = bwd_segs_.rbegin(); seg_it != bwd_segs_.rend(); ++seg_it) {
      const level_seg_t& seg = *seg_it;
      if (seg.bundled == 2) {
        const i_t base = head_level_ptr_[seg.lo];
        ldlt_detail::ldlt_backward_chain_kernel<i_t, f_t>
          <<<1, seg.block_dim, 0, stream>>>(d_head_level_cols_.data() + base,
                                            d_chain_win_.data() + seg.win_begin,
                                            seg.win_count,
                                            d_chain_bop_off_.data() + base,
                                            d_chain_col_slot_.data(),
                                            d_Lp_.data(),
                                            d_Li_.data(),
                                            d_Lx_.data(),
                                            d_work_.data());
        RAFT_CHECK_CUDA(stream);
        continue;
      }
      if (seg.bundled == 1) {
        ldlt_detail::ldlt_backward_bundle_kernel<i_t, f_t>
          <<<1, seg.block_dim, 0, stream>>>(d_head_level_ptr_.data(),
                                            d_head_level_cols_.data(),
                                            seg.lo,
                                            seg.hi,
                                            d_Lp_.data(),
                                            d_Li_.data(),
                                            d_Lx_.data(),
                                            d_work_.data());
        RAFT_CHECK_CUDA(stream);
        continue;
      }
      for (i_t l = seg.hi - 1; l >= seg.lo; l--) {
        const i_t level_start = head_level_ptr_[l];
        const i_t level_size  = head_level_ptr_[l + 1] - level_start;
        if (level_size == 0) { continue; }
        ldlt_detail::ldlt_backward_level_kernel<i_t, f_t>
          <<<level_size, block_dim, 0, stream>>>(d_head_level_cols_.data(),
                                                 level_start,
                                                 d_Lp_.data(),
                                                 d_Li_.data(),
                                                 d_Lx_.data(),
                                                 d_work_.data());
        RAFT_CHECK_CUDA(stream);
      }
    }
    if (tail_dim_ > 0) { RAFT_CUBLAS_TRY(cublasSetPointerMode(cublas, prev_mode)); }

    ldlt_detail::ldlt_perm_scatter_kernel<i_t, f_t><<<grid_n, block_dim, 0, stream>>>(
      d_work_.data(), d_perm_.data(), n_, scaling_active_ ? d_scale_.data() : nullptr, d_x);
    RAFT_CHECK_CUDA(stream);
  }

  // Triangular solves on the device. The level-scheduled sequence is launch
  // latency bound, so it is captured once into a CUDA graph against staging
  // buffers and replayed per solve (the scaling mode is part of the captured
  // state; analyze invalidates the graph). The first call after each
  // (re)capture trigger runs eagerly to warm library workspaces.
  i_t solve_values_device(const f_t* d_b, f_t* d_x, rmm::cuda_stream_view stream)
  {
    if (!factorized_) {
      settings_.log.printf("Solve called before factorize\n");
      return -1;
    }
    f_t t_solve_start = tic();
    if (solve_graph_valid_ && solve_graph_scaling_ != scaling_active_) {
      RAFT_CUDA_TRY(cudaGraphExecDestroy(solve_graph_));
      solve_graph_       = nullptr;
      solve_graph_valid_ = false;
      solve_graph_warm_  = false;
    }
    raft::copy(d_solve_b_.data(), d_b, n_, stream);
    if (!solve_graph_valid_) {
      if (!solve_graph_warm_) {
        solve_kernels_device(d_solve_b_.data(), d_solve_x_.data(), stream);
        solve_graph_warm_ = true;
      } else {
        RAFT_CUDA_TRY(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal));
        solve_kernels_device(d_solve_b_.data(), d_solve_x_.data(), stream);
        cudaGraph_t graph = nullptr;
        RAFT_CUDA_TRY(cudaStreamEndCapture(stream, &graph));
        RAFT_CUDA_TRY(cudaGraphInstantiate(&solve_graph_, graph, nullptr, nullptr, 0));
        RAFT_CUDA_TRY(cudaGraphDestroy(graph));
        solve_graph_valid_   = true;
        solve_graph_scaling_ = scaling_active_;
        RAFT_CUDA_TRY(cudaGraphLaunch(solve_graph_, stream));
      }
    } else {
      RAFT_CUDA_TRY(cudaGraphLaunch(solve_graph_, stream));
    }
    raft::copy(d_x, d_solve_x_.data(), n_, stream);

    // Drain the stream: the barrier's per-iteration host<->device traffic
    // uses pageable staging, and an ever-deepening async queue makes every
    // later cudaMemcpyAsync call block for ~25us (staging-ring exhaustion) -
    // measurably slower than the explicit synchronize.
    RAFT_CUDA_TRY(cudaStreamSynchronize(stream));
    if (stats_enabled_) { solve_time_ += toc(t_solve_start); }
    n_solve_calls_++;
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

  // Structured pivot handling (set_pivot_structure); -2 = legacy unsigned
  // static pivoting.
  i_t pivot_n_neg_{-2};
  f_t pivot_neg_floor_{0};
  f_t pivot_pos_floor_{0};
  i_t last_sign_corrections_{0};
  bool scaling_active_{false};
  bool stats_enabled_{false};

  // Captured solve sequence (see solve_values_device).
  cudaGraphExec_t solve_graph_{nullptr};
  bool solve_graph_valid_{false};
  bool solve_graph_warm_{false};
  bool solve_graph_scaling_{false};

  // Captured factorization sequence (see factorize_values_device).
  cudaGraphExec_t factor_graph_{nullptr};
  bool factor_graph_valid_{false};
  bool factor_graph_warm_{false};
  const f_t* factor_graph_values_{nullptr};
  f_t factor_graph_neg_floor_{0};
  f_t factor_graph_pos_floor_{0};
  i_t factor_graph_n_neg_{0};

  // Permutation: perm_[k] = original index of the k-th pivot.
  std::vector<i_t> perm_;
  std::vector<i_t> perm_inv_;

  // Elimination tree and level schedule.
  std::vector<i_t> etree_parent_;
  std::vector<i_t> level_ptr_;   // size n_levels_ + 1
  std::vector<i_t> level_cols_;  // columns grouped by level
  std::vector<i_t> head_level_cols_;
  std::vector<i_t> head_level_max_rp_;
  std::vector<i_t> tail_seg_;
  std::vector<i_t> rp_head_end_;
  std::vector<i_t> schur_light_;
  std::vector<i_t> schur_heavy_;

  // Exact factor column counts (including the diagonal), Gilbert-Ng-Peyton.
  std::vector<i_t> col_counts_;

  // Bundled level execution plans (see build_level_bundles).
  struct level_seg_t {
    i_t lo;
    i_t hi;
    i_t bundled;  // 0 = per-level, 1 = bundle, 2 = chain
    i_t block_dim;
    i_t win_begin{0};  // chain: index into chain_win_ (window starts + end)
    i_t win_count{0};  // chain: number of windows
  };
  std::vector<level_seg_t> factor_segs_;
  std::vector<level_seg_t> fwd_segs_;
  std::vector<level_seg_t> bwd_segs_;

  // Chain metadata (host mirrors; see ldlt_factor_chain_kernel).
  std::vector<i_t> chain_win_;       // concatenated per-run window start lists
  std::vector<i_t> chain_smoff_;     // per head position: window-relative data offset
  std::vector<i_t> chain_fop_off_;   // per head position: running in-window rp-entry prefix
  std::vector<i_t> chain_bop_off_;   // per head position: running in-window col-entry prefix
  std::vector<i_t> chain_rp_slot_;   // per rp entry: window slot or -1
  std::vector<i_t> chain_col_slot_;  // per column entry: window slot or -1

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
  std::vector<int64_t> a2l_;

  // Dense trailing block [tail_start_, n_): the top of the elimination tree
  // (long sequential chains, nearly dense columns) factored as one dense
  // LDL^T block. tail_dim_ == 0 disables the path.
  i_t tail_start_{0};
  i_t tail_dim_{0};
  i_t n_head_levels_{0};
  i_t schur_chunk_{0};
  i_t n_schur_light_{0};
  i_t n_schur_heavy_{0};
  std::vector<i_t> head_level_ptr_;

  // Numeric values (host mirror; the device factor lives in d_Lx_/d_D_).
  std::vector<f_t> Lx_;
  std::vector<f_t> D_;
  std::vector<f_t> work_;

  // Cumulative phase timings (CUOPT_LDLT_STATS).
  f_t factor_time_{0};
  f_t solve_time_{0};
  i_t n_factor_calls_{0};
  i_t n_solve_calls_{0};

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
  rmm::device_uvector<int64_t> d_a2l_;
  rmm::device_uvector<i_t> d_level_cols_;
  rmm::device_uvector<i_t> d_head_level_cols_;
  rmm::device_uvector<i_t> d_head_level_ptr_;
  rmm::device_uvector<i_t> d_tail_seg_;
  rmm::device_uvector<i_t> d_rp_head_end_;
  rmm::device_uvector<i_t> d_chain_win_;
  rmm::device_uvector<i_t> d_chain_smoff_;
  rmm::device_uvector<i_t> d_chain_fop_off_;
  rmm::device_uvector<i_t> d_chain_bop_off_;
  rmm::device_uvector<i_t> d_chain_rp_slot_;
  rmm::device_uvector<i_t> d_chain_col_slot_;
  rmm::device_uvector<i_t> d_schur_light_;
  rmm::device_uvector<i_t> d_schur_heavy_;
  rmm::device_uvector<f_t> d_dense_;
  rmm::device_uvector<f_t> d_panelV_;
  rmm::device_uvector<f_t> d_panelW_;
  rmm::device_uvector<i_t> d_perm_;
  rmm::device_uvector<f_t> d_Lx_;
  rmm::device_uvector<f_t> d_D_;
  rmm::device_uvector<f_t> d_a_values_;
  rmm::device_uvector<f_t> d_work_;
  rmm::device_uvector<f_t> d_diag0_;
  rmm::device_uvector<f_t> d_scale_;
  rmm::device_uvector<f_t> d_solve_b_;
  rmm::device_uvector<f_t> d_solve_x_;
  rmm::device_scalar<i_t> d_fail_;
  rmm::device_scalar<i_t> d_static_pivots_;
  rmm::device_scalar<i_t> d_sign_corrections_;
};

}  // namespace cuopt::linear_programming::dual_simplex
