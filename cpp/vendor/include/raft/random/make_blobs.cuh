/*
 * cuOpt vendored RAFT shim — make_blobs (synthetic clustered points).
 *
 * Legacy pointer-form signature used by src/routing/generator/generator.cu.
 * Functionally generates Gaussian blobs around random centers; NOT bit-identical
 * to upstream raft, so generator golden data may need regeneration.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <raft/random/rng_device.cuh>
#include <raft/util/cudart_utils.hpp>

#include <cstdint>

namespace raft::random {

namespace detail {

// out is n_rows x n_cols, column-major when row_major == false (the cuOpt case).
template <typename DataT, typename IdxT, typename LabelT>
__global__ void make_blobs_kernel(DataT* out,
                                  LabelT* labels,
                                  IdxT n_rows,
                                  IdxT n_cols,
                                  IdxT n_clusters,
                                  bool row_major,
                                  DataT cluster_std,
                                  DataT center_min,
                                  DataT center_max,
                                  uint64_t seed)
{
  for (IdxT row = static_cast<IdxT>(blockIdx.x) * blockDim.x + threadIdx.x; row < n_rows;
       row += static_cast<IdxT>(gridDim.x) * blockDim.x) {
    raft::random::PCGenerator gen(seed, static_cast<uint64_t>(row), 0);
    uint32_t cl;
    gen.next(cl);
    IdxT cluster = static_cast<IdxT>(cl % static_cast<uint32_t>(n_clusters));
    if (labels != nullptr) { labels[row] = static_cast<LabelT>(cluster); }

    // Deterministic per-cluster center drawn from a cluster-seeded generator.
    for (IdxT c = 0; c < n_cols; ++c) {
      raft::random::PCGenerator cgen(
        seed ^ 0x9e3779b97f4a7c15ULL, static_cast<uint64_t>(cluster) * n_cols + c, 0);
      DataT u;
      cgen.next(u);
      DataT center = center_min + u * (center_max - center_min);

      DataT noise;
      gen.next(noise);  // [0,1); cheap pseudo-gaussian offset
      DataT val = center + (noise - DataT(0.5)) * DataT(2) * cluster_std;

      IdxT idx = row_major ? (row * n_cols + c) : (c * n_rows + row);
      out[idx] = val;
    }
  }
}

}  // namespace detail

/** @brief Generate n_rows points (n_cols each) clustered around n_clusters centers. */
template <typename DataT, typename IdxT, typename LabelT>
void make_blobs(DataT* out,
                LabelT* labels,
                IdxT n_rows,
                IdxT n_cols,
                IdxT n_clusters,
                cudaStream_t stream,
                bool row_major                = true,
                const DataT* /*centers*/      = nullptr,
                const DataT* /*cluster_std_v*/ = nullptr,
                DataT cluster_std             = DataT(1),
                bool /*shuffle*/              = true,
                DataT center_box_min          = DataT(-10),
                DataT center_box_max          = DataT(10),
                uint64_t seed                 = 0ULL)
{
  if (n_rows <= 0 || n_cols <= 0) { return; }
  constexpr int kThreads = 256;
  int const blocks       = static_cast<int>((n_rows + kThreads - 1) / kThreads);
  detail::make_blobs_kernel<<<blocks < 65535 ? blocks : 65535, kThreads, 0, stream>>>(
    out, labels, n_rows, n_cols, n_clusters, row_major, cluster_std, center_box_min, center_box_max,
    seed);
  RAFT_CHECK_CUDA(stream);
}

}  // namespace raft::random
