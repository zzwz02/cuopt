/*
 * cuOpt vendored RMM shim — owning CUDA stream (RAII).
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <rmm/cuda_stream_view.hpp>
#include <rmm/detail/error.hpp>

#include <cuda_runtime_api.h>

#include <memory>

namespace rmm {

/** @brief Owning wrapper for a non-blocking cudaStream_t. */
class cuda_stream {
 public:
  enum class flags : unsigned int {
    default_flags = cudaStreamDefault,
    non_blocking  = cudaStreamNonBlocking,
  };

  cuda_stream(cuda_stream&&)                 = default;
  cuda_stream& operator=(cuda_stream&&)      = default;
  cuda_stream(cuda_stream const&)            = delete;
  cuda_stream& operator=(cuda_stream const&) = delete;
  ~cuda_stream()                             = default;

  explicit cuda_stream(flags stream_flags = flags::non_blocking)
    : stream_{[stream_flags]() {
                auto* str = new cudaStream_t;
                RMM_CUDA_TRY(
                  cudaStreamCreateWithFlags(str, static_cast<unsigned int>(stream_flags)));
                return str;
              }(),
              [](cudaStream_t* str) {
                RMM_ASSERT(str != nullptr, "Invalid stream.");
                cudaStreamDestroy(*str);
                delete str;
              }}
  {
  }

  [[nodiscard]] bool is_valid() const { return stream_ != nullptr; }
  [[nodiscard]] cudaStream_t value() const
  {
    RMM_ASSERT(stream_ != nullptr, "Invalid stream.");
    return *stream_;
  }
  // NOLINTNEXTLINE(google-explicit-constructor)
  [[nodiscard]] operator cudaStream_t() const noexcept { return value(); }
  [[nodiscard]] cuda_stream_view view() const { return cuda_stream_view{value()}; }
  // NOLINTNEXTLINE(google-explicit-constructor)
  [[nodiscard]] operator cuda_stream_view() const { return view(); }

  void synchronize() const { RMM_CUDA_TRY(cudaStreamSynchronize(value())); }
  void synchronize_no_throw() const noexcept
  {
    if (cudaStreamSynchronize(value()) != cudaSuccess) { cudaGetLastError(); }
  }

 private:
  std::unique_ptr<cudaStream_t, std::function<void(cudaStream_t*)>> stream_;
};

}  // namespace rmm
