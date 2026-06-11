/*
 * cuOpt vendored RAFT shim — device_span. SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <raft/core/span.hpp>
#include <raft/util/cudart_utils.hpp>  // raft::copy, transitively expected by span users

#include <cstddef>

namespace raft {

/** @brief A span of device-accessible memory. */
template <typename T, std::size_t Extent = dynamic_extent>
using device_span = span<T, true, Extent>;

}  // namespace raft
