/*
 * cuOpt vendored RMM shim. SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <rmm/detail/error.hpp>

#include <cuda_runtime_api.h>

#include <cstddef>
#include <utility>

namespace rmm {

/** Strong type for a CUDA device id. */
struct cuda_device_id {
  using value_type = int;

  constexpr cuda_device_id() noexcept = default;
  constexpr explicit cuda_device_id(value_type dev_id) noexcept : id_{dev_id} {}

  [[nodiscard]] constexpr value_type value() const noexcept { return id_; }

  constexpr bool operator==(cuda_device_id const& other) const noexcept { return id_ == other.id_; }
  constexpr bool operator!=(cuda_device_id const& other) const noexcept { return id_ != other.id_; }

 private:
  value_type id_{0};
};

/** @brief Returns the current CUDA device id. */
[[nodiscard]] inline cuda_device_id get_current_cuda_device()
{
  cuda_device_id::value_type dev_id{0};
  RMM_CUDA_TRY(cudaGetDevice(&dev_id));
  return cuda_device_id{dev_id};
}

/** @brief Returns the number of visible CUDA devices. */
[[nodiscard]] inline int get_num_cuda_devices()
{
  cuda_device_id::value_type num_dev{0};
  RMM_CUDA_TRY(cudaGetDeviceCount(&num_dev));
  return num_dev;
}

/** @brief Returns (free, total) device memory in bytes for the current device. */
[[nodiscard]] inline std::pair<std::size_t, std::size_t> available_device_memory()
{
  std::size_t free{}, total{};
  RMM_CUDA_TRY(cudaMemGetInfo(&free, &total));
  return {free, total};
}

/** @brief RAII helper that sets the active CUDA device and restores it on destruction. */
struct cuda_set_device_raii {
  explicit cuda_set_device_raii(cuda_device_id dev_id)
    : old_device_{get_current_cuda_device()},
      needs_reset_{dev_id.value() >= 0 && old_device_ != dev_id}
  {
    if (needs_reset_) { RMM_CUDA_TRY(cudaSetDevice(dev_id.value())); }
  }

  ~cuda_set_device_raii() noexcept
  {
    if (needs_reset_) { cudaSetDevice(old_device_.value()); }
  }

  cuda_set_device_raii(cuda_set_device_raii const&)            = delete;
  cuda_set_device_raii& operator=(cuda_set_device_raii const&) = delete;
  cuda_set_device_raii(cuda_set_device_raii&&)                 = delete;
  cuda_set_device_raii& operator=(cuda_set_device_raii&&)      = delete;

 private:
  cuda_device_id old_device_;
  bool needs_reset_;
};

}  // namespace rmm
