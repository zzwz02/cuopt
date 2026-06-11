/*
 * cuOpt vendored RMM shim. SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <rmm/detail/error.hpp>

#include <cuda_runtime_api.h>

#include <atomic>
#include <cstddef>
#include <cstdint>

namespace rmm {

/**
 * @brief Strongly-typed non-owning wrapper for a cudaStream_t.
 */
class cuda_stream_view {
 public:
  constexpr cuda_stream_view()                                   = default;
  constexpr cuda_stream_view(cuda_stream_view const&)            = default;
  constexpr cuda_stream_view(cuda_stream_view&&)                 = default;
  constexpr cuda_stream_view& operator=(cuda_stream_view const&) = default;
  constexpr cuda_stream_view& operator=(cuda_stream_view&&)      = default;
  ~cuda_stream_view()                                            = default;

  // Disallow implicit construction from an int (e.g. 0/NULL) to catch mistakes.
  constexpr cuda_stream_view(int)            = delete;
  constexpr cuda_stream_view(std::nullptr_t) = delete;

  // NOLINTNEXTLINE(google-explicit-constructor)
  constexpr cuda_stream_view(cudaStream_t stream) noexcept : stream_{stream} {}

  [[nodiscard]] constexpr cudaStream_t value() const noexcept { return stream_; }

  // NOLINTNEXTLINE(google-explicit-constructor)
  constexpr operator cudaStream_t() const noexcept { return value(); }

  [[nodiscard]] inline bool is_per_thread_default() const noexcept;
  [[nodiscard]] inline bool is_default() const noexcept;

  void synchronize() const { RMM_CUDA_TRY(cudaStreamSynchronize(stream_)); }

  void synchronize_no_throw() const noexcept
  {
    auto const status = cudaStreamSynchronize(stream_);
    if (status != cudaSuccess) { cudaGetLastError(); }
  }

 private:
  cudaStream_t stream_{};
};

// Stream constants. cuda_stream_legacy == cudaStreamLegacy, etc.
static constexpr cuda_stream_view cuda_stream_default{};
static const cuda_stream_view cuda_stream_legacy{cudaStreamLegacy};
static const cuda_stream_view cuda_stream_per_thread{cudaStreamPerThread};

[[nodiscard]] inline bool cuda_stream_view::is_per_thread_default() const noexcept
{
#ifdef CUDA_API_PER_THREAD_DEFAULT_STREAM
  return value() == cuda_stream_per_thread || value() == nullptr;
#else
  return value() == cuda_stream_per_thread;
#endif
}

[[nodiscard]] inline bool cuda_stream_view::is_default() const noexcept
{
#ifdef CUDA_API_PER_THREAD_DEFAULT_STREAM
  return value() == cuda_stream_legacy;
#else
  return value() == cuda_stream_legacy || value() == nullptr;
#endif
}

[[nodiscard]] inline bool operator==(cuda_stream_view lhs, cuda_stream_view rhs)
{
  return lhs.value() == rhs.value();
}

[[nodiscard]] inline bool operator!=(cuda_stream_view lhs, cuda_stream_view rhs)
{
  return !(lhs == rhs);
}

}  // namespace rmm
