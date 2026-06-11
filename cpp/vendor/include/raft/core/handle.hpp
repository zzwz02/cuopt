/*
 * cuOpt vendored RAFT shim — handle_t (stream + lazy cuBLAS/cuSPARSE).
 * cuda::mr-free. SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <raft/core/cublas_macros.hpp>
#include <raft/core/cusparse_macros.hpp>
#include <raft/core/device_mdspan.hpp>  // raft::make_mdspan/extents transitively expected
#include <raft/util/cuda_rt_essentials.hpp>
#include <raft/util/cudart_utils.hpp>  // raft::copy transitively expected via handle users

#include <rmm/cuda_stream_view.hpp>
#include <rmm/exec_policy.hpp>

#include <cublas_v2.h>
#include <cusparse.h>

#include <memory>
#include <type_traits>

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
  // Default to the per-thread default stream, matching raft::handle_t. The
  // legacy null stream (cuda_stream_default) has global-synchronization
  // semantics and cannot be the target of CUDA graph capture, which cuOpt's
  // manual_cuda_graph_t relies on; cudaStreamPerThread is capturable.
  handle_t() : handle_t(rmm::cuda_stream_per_thread) {}

  explicit handle_t(rmm::cuda_stream_view stream)
    : stream_view_{stream},
      thrust_policy_{std::make_unique<rmm::exec_policy>(stream)}
  {
  }

  // Copyable like raft::handle_t, which copies SHARE the underlying cuBLAS /
  // cuSPARSE resources (ref-counted). This preserves handle state — crucially
  // the device pointer mode cuOpt sets via init_handler — across copies, and
  // ref-counts destruction so no double-free occurs. The copy gets a fresh
  // Thrust policy bound to the same stream. Move is deleted, matching raft.
  handle_t(handle_t const& other)
    : stream_view_{other.stream_view_},
      thrust_policy_{std::make_unique<rmm::exec_policy>(other.stream_view_)},
      cublas_handle_{other.cublas_handle_},
      cusparse_handle_{other.cusparse_handle_}
  {
  }
  handle_t& operator=(handle_t const&) = delete;
  handle_t(handle_t&&)                 = delete;
  handle_t& operator=(handle_t&&)      = delete;

  // Minimal comms surface (cuOpt is single-GPU here; comms is never initialized).
  struct comms_t {
    void barrier() const {}
  };
  [[nodiscard]] bool comms_initialized() const noexcept { return false; }
  [[nodiscard]] comms_t get_comms() const noexcept { return {}; }

  virtual ~handle_t() = default;

  [[nodiscard]] rmm::cuda_stream_view get_stream() const noexcept { return stream_view_; }

  [[nodiscard]] rmm::exec_policy& get_thrust_policy() const noexcept { return *thrust_policy_; }

  void sync_stream() const { stream_view_.synchronize(); }
  void sync_stream(rmm::cuda_stream_view stream) const { stream.synchronize(); }

  // NOTE: the stream is bound to the cuBLAS/cuSPARSE handle only at creation and
  // whenever set_cuda_stream() changes it — NOT on every accessor call. cuOpt
  // captures these calls inside CUDA graphs (manual_cuda_graph_t), and calling
  // the stateful cublasSetStream/cusparseSetStream during stream capture breaks
  // the captured graph (cuBLAS execution fails on replay).
  [[nodiscard]] cublasHandle_t get_cublas_handle() const
  {
    if (!cublas_handle_) {
      cublasHandle_t h{nullptr};
      RAFT_CUBLAS_TRY(cublasCreate(&h));
      RAFT_CUBLAS_TRY(cublasSetStream(h, stream_view_.value()));
      cublas_handle_ = std::shared_ptr<std::remove_pointer_t<cublasHandle_t>>(
        h, [](cublasHandle_t p) { if (p != nullptr) { cublasDestroy(p); } });
    }
    return cublas_handle_.get();
  }

  [[nodiscard]] cusparseHandle_t get_cusparse_handle() const
  {
    if (!cusparse_handle_) {
      cusparseHandle_t h{nullptr};
      RAFT_CUSPARSE_TRY(cusparseCreate(&h));
      RAFT_CUSPARSE_TRY(cusparseSetStream(h, stream_view_.value()));
      cusparse_handle_ = std::shared_ptr<std::remove_pointer_t<cusparseHandle_t>>(
        h, [](cusparseHandle_t p) { if (p != nullptr) { cusparseDestroy(p); } });
    }
    return cusparse_handle_.get();
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
    if (cublas_handle_) { RAFT_CUBLAS_TRY(cublasSetStream(cublas_handle_.get(), stream.value())); }
    if (cusparse_handle_) {
      RAFT_CUSPARSE_TRY(cusparseSetStream(cusparse_handle_.get(), stream.value()));
    }
  }

 private:
  rmm::cuda_stream_view stream_view_{rmm::cuda_stream_default};
  mutable std::unique_ptr<rmm::exec_policy> thrust_policy_;
  mutable std::shared_ptr<std::remove_pointer_t<cublasHandle_t>> cublas_handle_;
  mutable std::shared_ptr<std::remove_pointer_t<cusparseHandle_t>> cusparse_handle_;
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
