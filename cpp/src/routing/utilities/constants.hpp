/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#pragma once

#include <raft/util/warp_constants.hpp>

namespace cuopt {
namespace routing {
namespace detail {

// Compile-time, platform-dependent (32 on NVIDIA, 64 on AMD wave64).
constexpr int warp_size  = raft::WarpSize;
constexpr int I_HALF_MAX = 65504;  // highest positive value representable by half
                                   // there is no numeric_limits implementation of half
}  // namespace detail
}  // namespace routing
}  // namespace cuopt
