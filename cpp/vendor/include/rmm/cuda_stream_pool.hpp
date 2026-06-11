/*
 * cuOpt vendored RMM shim — fixed-size pool of non-blocking CUDA streams.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <rmm/cuda_stream.hpp>
#include <rmm/cuda_stream_view.hpp>
#include <rmm/detail/error.hpp>

#include <atomic>
#include <cstddef>
#include <memory>
#include <vector>

namespace rmm {

class cuda_stream_pool {
 public:
  static constexpr std::size_t default_size{16};

  explicit cuda_stream_pool(std::size_t pool_size = default_size,
                            cuda_stream::flags = cuda_stream::flags::non_blocking)
    : streams_(pool_size)
  {
    RMM_EXPECTS(pool_size > 0, "Stream pool size must be greater than zero");
  }

  cuda_stream_pool(cuda_stream_pool&&)                 = delete;
  cuda_stream_pool(cuda_stream_pool const&)            = delete;
  cuda_stream_pool& operator=(cuda_stream_pool&&)      = delete;
  cuda_stream_pool& operator=(cuda_stream_pool const&) = delete;
  ~cuda_stream_pool()                                  = default;

  [[nodiscard]] rmm::cuda_stream_view get_stream() const noexcept
  {
    return streams_[(next_stream_++) % streams_.size()].view();
  }

  [[nodiscard]] rmm::cuda_stream_view get_stream(std::size_t stream_id) const
  {
    return streams_[stream_id % streams_.size()].view();
  }

  [[nodiscard]] std::size_t get_pool_size() const noexcept { return streams_.size(); }

 private:
  std::vector<rmm::cuda_stream> streams_;
  mutable std::atomic_size_t next_stream_{};
};

}  // namespace rmm
