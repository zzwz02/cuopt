/*
 * cuOpt vendored RAFT shim — NVTX scoped ranges (no-op).
 * Profiling-only; a no-op keeps behavior identical without the NVTX dependency.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <utility>

namespace raft::common::nvtx {

namespace domain {
struct global {};
}  // namespace domain

template <typename Domain = domain::global>
struct range {
  template <typename... Args>
  explicit range(Args&&...)
  {
  }
  ~range()                       = default;
  range(range const&)            = delete;
  range& operator=(range const&) = delete;
  range(range&&)                 = delete;
  range& operator=(range&&)      = delete;
};

// Allow `raft::common::nvtx::range scope("name", ...)` without explicit domain.
template <typename... Args>
range(Args&&...) -> range<domain::global>;

template <typename Domain = domain::global, typename... Args>
inline void push_range(Args&&...)
{
}

template <typename Domain = domain::global>
inline void pop_range()
{
}

}  // namespace raft::common::nvtx
