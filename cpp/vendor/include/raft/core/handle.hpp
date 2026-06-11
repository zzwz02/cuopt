/*
 * cuOpt vendored RAFT shim — handle_t (stream + lazy cuBLAS/cuSPARSE).
 * cuda::mr-free. SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <raft/core/cublas_macros.hpp>
#include <raft/core/cusparse_macros.hpp>
#include <raft/core/device_mdspan.hpp>  // raft::make_mdspan/extents transitively expected
#include <raft/util/cuda_rt_essentials.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/exec_policy.hpp>

#include <cublas_v2.h>
#include <cusparse.h>

#include <memory>

namespace raft {

/**
 * @brief Minimal replacement for raft::handle_t.
 *
 * Owns the working stream, a Thrust execution policy bound to it (drawing temp
 * storage from the cuOpt device pool), and lazily-created cuBLAS / cuSPARSE
 * handles kept bound to the current stream.
 */
class handle_t {
 public:
  handle_t() : handle_t(rmm::cuda_stream_default) {}

  explicit handle_t(rmm::cuda_stream_view stream)
    : stream_view_{stream},
      thrust_policy_{std::make_unique<rmm::exec_policy>(stream)}
  {
  }

  handle_t(handle_t const&)            = delete;
  handle_t& operator=(handle_t const&) = delete;
  handle_t(handle_t&&)                 = delete;
  handle_t& operator=(handle_t&&)      = delete;

  virtual ~handle_t()
  {
    if (cublas_handle_ != nullptr) { cublasDestroy(cublas_handle_); }
    if (cusparse_handle_ != nullptr) { cusparseDestroy(cusparse_handle_); }
  }

  [[nodiscard]] rmm::cuda_stream_view get_stream() const noexcept { return stream_view_; }

  [[nodiscard]] rmm::exec_policy& get_thrust_policy() const noexcept { return *thrust_policy_; }

  void sync_stream() const { stream_view_.synchronize(); }
  void sync_stream(rmm::cuda_stream_view stream) const { stream.synchronize(); }

  [[nodiscard]] cublasHandle_t get_cublas_handle() const
  {
    if (cublas_handle_ == nullptr) { RAFT_CUBLAS_TRY(cublasCreate(&cublas_handle_)); }
    RAFT_CUBLAS_TRY(cublasSetStream(cublas_handle_, stream_view_.value()));
    return cublas_handle_;
  }

  [[nodiscard]] cusparseHandle_t get_cusparse_handle() const
  {
    if (cusparse_handle_ == nullptr) { RAFT_CUSPARSE_TRY(cusparseCreate(&cusparse_handle_)); }
    RAFT_CUSPARSE_TRY(cusparseSetStream(cusparse_handle_, stream_view_.value()));
    return cusparse_handle_;
  }

  [[nodiscard]] int get_device() const
  {
    int dev{0};
    RAFT_CUDA_TRY(cudaGetDevice(&dev));
    return dev;
  }

  [[nodiscard]] cudaDeviceProp const& get_device_properties() const
  {
    if (!device_prop_valid_) {
      RAFT_CUDA_TRY(cudaGetDeviceProperties(&device_prop_, get_device()));
      device_prop_valid_ = true;
    }
    return device_prop_;
  }

  void set_cuda_stream(rmm::cuda_stream_view stream)
  {
    stream_view_   = stream;
    thrust_policy_ = std::make_unique<rmm::exec_policy>(stream);
    if (cublas_handle_ != nullptr) { RAFT_CUBLAS_TRY(cublasSetStream(cublas_handle_, stream.value())); }
    if (cusparse_handle_ != nullptr) {
      RAFT_CUSPARSE_TRY(cusparseSetStream(cusparse_handle_, stream.value()));
    }
  }

 private:
  rmm::cuda_stream_view stream_view_{rmm::cuda_stream_default};
  mutable std::unique_ptr<rmm::exec_policy> thrust_policy_;
  mutable cublasHandle_t cublas_handle_{nullptr};
  mutable cusparseHandle_t cusparse_handle_{nullptr};
  mutable cudaDeviceProp device_prop_{};
  mutable bool device_prop_valid_{false};
};

// raft uses `resources` as the base type name; alias both to handle_t.
using resources        = handle_t;
using device_resources = handle_t;

namespace resource {

inline rmm::cuda_stream_view get_cuda_stream(handle_t const& handle) { return handle.get_stream(); }

inline void set_cuda_stream(handle_t& handle, rmm::cuda_stream_view stream)
{
  handle.set_cuda_stream(stream);
}

inline void sync_stream(handle_t const& handle) { handle.sync_stream(); }

inline int get_device_id(handle_t const& handle) { return handle.get_device(); }

inline cublasHandle_t get_cublas_handle(handle_t const& handle)
{
  return handle.get_cublas_handle();
}

inline cusparseHandle_t get_cusparse_handle(handle_t const& handle)
{
  return handle.get_cusparse_handle();
}

}  // namespace resource

}  // namespace raft
