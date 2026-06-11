/*
 * cuOpt vendored RAFT shim — host_span. SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <raft/core/span.hpp>

#include <cstddef>

namespace raft {

/** @brief A span of host-accessible memory. */
template <typename T, std::size_t Extent = dynamic_extent>
using host_span = span<T, false, Extent>;

}  // namespace raft
