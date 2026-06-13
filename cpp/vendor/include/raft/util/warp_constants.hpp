/*
 * cuOpt vendored RAFT shim — compile-time warp/wavefront constants.
 *
 * Host-safe (no device code): includable from both g++ host TUs and device
 * code. Everything here is constexpr so warp-size-dependent array bounds,
 * masks and loop trip counts fold at compile time on every platform —
 * 32 lanes on NVIDIA, 64 lanes on AMD CDNA (e.g. MI100, wave64-only).
 *
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <cstdint>
#include <type_traits>

namespace raft {

#if defined(__AMDGCN_WAVEFRONT_SIZE__)
static constexpr int WarpSize = __AMDGCN_WAVEFRONT_SIZE__;
#elif defined(__HIP_PLATFORM_AMD__) || defined(__MACA__) || defined(__MACACC__) || \
  defined(__MACA_ARCH__) || defined(__XCORE_WN__)
static constexpr int WarpSize = 64;  // MACA C500/CDNA-style targets are wave64.
#else
static constexpr int WarpSize = 32;
#endif

static_assert(WarpSize == 32 || WarpSize == 64, "unsupported warp size");

/** log2(WarpSize), for shift-based lane/warp index math. */
static constexpr int WarpSizeLog2 = (WarpSize == 64) ? 6 : 5;

/**
 * @brief Lane mask: one bit per lane of a warp/wavefront.
 *
 * 32-bit on NVIDIA, 64-bit on wave64 — every variable holding a
 * ballot/activemask result must use this type, never a fixed uint32_t.
 */
using lane_mask_t = std::conditional_t<(WarpSize > 32), unsigned long long, unsigned int>;

static_assert(sizeof(lane_mask_t) * 8 == WarpSize, "lane_mask_t must have one bit per lane");

/** All-lanes mask (replaces hardcoded 0xffffffff). */
static constexpr lane_mask_t LANE_MASK_ALL = static_cast<lane_mask_t>(~0ull);

}  // namespace raft
