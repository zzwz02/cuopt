/*
 * cuOpt vendored RAFT shim — scoped CUDA device setter.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <raft/util/cudart_utils.hpp>

#include <cuda_runtime_api.h>

namespace raft {

/** @brief RAII helper that sets the active CUDA device and restores it. */
struct device_setter {
  explicit device_setter(int new_device) : prev_device_{get_current_device()}
  {
    RAFT_CUDA_TRY(cudaSetDevice(new_device));
  }

  ~device_setter() { RAFT_CUDA_TRY_NO_THROW(cudaSetDevice(prev_device_)); }

  device_setter(device_setter const&)            = delete;
  device_setter& operator=(device_setter const&) = delete;
  device_setter(device_setter&&)                 = delete;
  device_setter& operator=(device_setter&&)      = delete;

  static int get_current_device()
  {
    int dev{0};
    RAFT_CUDA_TRY(cudaGetDevice(&dev));
    return dev;
  }

  static int get_device_count()
  {
    int count{0};
    RAFT_CUDA_TRY(cudaGetDeviceCount(&count));
    return count;
  }

 private:
  int prev_device_;
};

}  // namespace raft
