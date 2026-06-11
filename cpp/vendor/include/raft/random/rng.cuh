/*
 * cuOpt vendored RAFT shim — host launchers for legacy pointer-form RNG.
 *
 * Provides raft::random::uniform / uniformInt over the vendored PCGenerator.
 * NOTE: functionally uniform but NOT bit-identical to upstream raft's kernels,
 * so synthetic data produced by src/routing/generator/generator.cu differs;
 * routing-generator golden expectations may need regeneration.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <raft/random/rng_device.cuh>
#include <raft/random/rng_state.hpp>
#include <raft/util/cudart_utils.hpp>

#include <cstdint>
#include <type_traits>

namespace raft::random {

namespace detail {

template <typename OutType, typename LenType>
__global__ void uniform_real_kernel(
  OutType* ptr, LenType len, OutType start, OutType range, uint64_t seed)
{
  for (LenType i = static_cast<LenType>(blockIdx.x) * blockDim.x + threadIdx.x; i < len;
       i += static_cast<LenType>(gridDim.x) * blockDim.x) {
    raft::random::PCGenerator gen(seed, static_cast<uint64_t>(i), 0);
    OutType val;
    gen.next(val);  // [0, 1)
    ptr[i] = start + val * range;
  }
}

template <typename OutType, typename LenType>
__global__ void uniform_int_kernel(
  OutType* ptr, LenType len, OutType start, uint64_t span, uint64_t seed)
{
  for (LenType i = static_cast<LenType>(blockIdx.x) * blockDim.x + threadIdx.x; i < len;
       i += static_cast<LenType>(gridDim.x) * blockDim.x) {
    raft::random::PCGenerator gen(seed, static_cast<uint64_t>(i), 0);
    uint32_t u;
    gen.next(u);
    ptr[i] = start + static_cast<OutType>(u % span);
  }
}

template <typename LenType>
inline int blocks_for(LenType len)
{
  constexpr int kThreads = 256;
  LenType b              = (len + kThreads - 1) / kThreads;
  return static_cast<int>(b < 65535 ? b : 65535);
}

}  // namespace detail

/** @brief Fill ptr[0..len) with uniform reals in [start, end). */
template <typename OutType, typename LenType>
void uniform(RngState& r, OutType* ptr, LenType len, OutType start, OutType end, cudaStream_t stream)
{
  static_assert(std::is_floating_point_v<OutType>, "uniform requires a floating-point type");
  if (len <= 0) { return; }
  detail::uniform_real_kernel<<<detail::blocks_for(len), 256, 0, stream>>>(
    ptr, len, start, end - start, r.seed);
  r.seed += static_cast<uint64_t>(len);
  RAFT_CHECK_CUDA(stream);
}

/** @brief Fill ptr[0..len) with uniform integers in [start, end). */
template <typename OutType, typename LenType>
void uniformInt(
  RngState& r, OutType* ptr, LenType len, OutType start, OutType end, cudaStream_t stream)
{
  static_assert(std::is_integral_v<OutType>, "uniformInt requires an integral type");
  if (len <= 0) { return; }
  uint64_t const span = static_cast<uint64_t>(end - start);
  detail::uniform_int_kernel<<<detail::blocks_for(len), 256, 0, stream>>>(
    ptr, len, start, span == 0 ? 1 : span, r.seed);
  r.seed += static_cast<uint64_t>(len);
  RAFT_CHECK_CUDA(stream);
}

}  // namespace raft::random
