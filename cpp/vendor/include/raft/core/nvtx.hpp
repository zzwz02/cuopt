/*
 * cuOpt vendored RAFT shim — NVTX scoped ranges.
 *
 * Default (no CUOPT_ENABLE_NVTX): no-op. Keeps the RAPIDS-free baseline free of any
 * NVTX dependency; behavior is identical to not having NVTX at all.
 *
 * With -DCUOPT_ENABLE_NVTX: emit real NVTX ranges via NVIDIA's header-only nvtx3
 * (<nvtx3/nvToolsExt.h>). RAII — pushes a named range on construction, pops on
 * destruction. On MACA this same call maps to MetaX mctx (/opt/maca/include/mctx)
 * through cu-bridge's compile-time remap, so the shim is not NVIDIA-only. The API
 * surface is unchanged from the no-op version, so it is a drop-in for every existing
 * `raft::common::nvtx::range`/`push_range`/`pop_range` call site.
 *
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <utility>

#if defined(CUOPT_ENABLE_NVTX)
#include <nvtx3/nvToolsExt.h>
#endif

namespace raft::common::nvtx {

namespace domain {
struct global {};
}  // namespace domain

#if defined(CUOPT_ENABLE_NVTX)

template <typename Domain = domain::global>
struct range {
  template <typename... Args>
  explicit range(const char* name, Args&&...)
  {
    nvtxRangePushA(name);
  }
  ~range() { nvtxRangePop(); }
  range(range const&)            = delete;
  range& operator=(range const&) = delete;
  range(range&&)                 = delete;
  range& operator=(range&&)      = delete;
};

// Allow `raft::common::nvtx::range scope("name", ...)` without explicit domain.
template <typename... Args>
range(Args&&...) -> range<domain::global>;

template <typename Domain = domain::global>
inline void push_range(const char* name)
{
  nvtxRangePushA(name);
}

template <typename Domain = domain::global>
inline void pop_range()
{
  nvtxRangePop();
}

#else

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

#endif

}  // namespace raft::common::nvtx
