/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#pragma once

#include <utilities/macros.cuh>

#include <thrust/host_vector.h>
#include <thrust/tuple.h>
#include <mutex>
#include <raft/core/device_span.hpp>
#include <raft/util/cuda_utils.cuh>
#include <raft/util/cudart_utils.hpp>
#include <rmm/device_uvector.hpp>
#include <shared_mutex>
#include <unordered_map>

namespace cuopt {

#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 700)
#error "cuOpt is only supported on Volta and newer architectures"
#endif

/** helper macro for device inlined functions */
#define DI  inline __device__
#define HDI inline __host__ __device__
#define HD  __host__ __device__

/**
 * For Pascal independent thread scheduling is not supported so we are using a seperate
 * add version. This version will return when there are duplicates instead of
 * udapting the key with the min value. Another approach would be to use a 64 bit
 * representation for values and predecessors and use atomicMin. This comes with
 * accuracy trade-offs. Hence the seperate add function for Pascal.
 **/
template <typename i_t>
DI bool acquire_lock(i_t* lock)
{
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 700)
  auto res = atomicCAS(lock, 0, 1);
  __threadfence();
  return res == 0;
#else
  while (atomicCAS(lock, 0, 1)) {
    __nanosleep(100);
  }
  __threadfence();
  return true;
#endif
}

template <typename i_t>
DI void release_lock(i_t* lock)
{
  __threadfence();
  atomicExch(lock, 0);
}

/**
 * Lock-step-SIMT-safe critical section.
 *
 * The idiom `acquire_lock(L); <critical>; release_lock(L);` deadlocks on GPUs
 * without independent thread scheduling (ITS) -- e.g. MetaX C500 / warp64 -- when
 * two or more lanes of the same warp contend for the same lock `L`. The lane that
 * wins the CAS exits acquire_lock()'s internal spin loop and parks at the loop's
 * reconvergence point while the losing lanes keep spinning; because the warp runs
 * in lock-step the winner can never advance to release_lock(), so the warp hangs
 * forever (no error, just a livelock at 100% util). NVIDIA Volta+ avoids this only
 * because ITS lets the winner make independent forward progress.
 *
 * with_lock() instead performs the critical section AND the release *inside* the
 * winning-CAS branch, before the loop's back-edge, so a lane that holds the lock
 * never waits on a same-warp lane that is still spinning. This is correct on every
 * architecture (with or without ITS).
 *
 * `critical_section` is invoked while the lock is held; it must not transfer
 * control out of itself (no return/break/continue targeting the caller). For
 * critical sections that need to `return` from the enclosing function, inline the
 * `while (!done) { if (atomicCAS(L,0,1)==0) { ...; atomicExch(L,0); done=true; } }`
 * pattern by hand instead.
 */
template <typename i_t, typename critical_fn_t>
DI void with_lock(i_t* lock, critical_fn_t&& critical_section)
{
  bool done = false;
  while (!done) {
    if (atomicCAS(lock, 0, 1) == 0) {
      __threadfence();
      critical_section();
      __threadfence();
      atomicExch(lock, 0);
      done = true;
    }
  }
}

template <typename i_t>
DI bool try_acquire_lock_block(i_t* lock)
{
  auto res = atomicCAS_block(lock, 0, 1);
  __threadfence_block();
  return res == 0;
}

template <typename i_t>
DI bool acquire_lock_block(i_t* lock)
{
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 700)
  return try_acquire_lock_block(lock);
#else
  while (atomicCAS_block(lock, 0, 1)) {
    __nanosleep(100);
  }
  __threadfence_block();
  return true;
#endif
}

template <typename i_t>
DI void release_lock_block(i_t* lock)
{
  __threadfence_block();
  atomicExch_block(lock, 0);
}

/**
 * Block-scope counterpart of with_lock() (uses atomicCAS_block / atomicExch_block).
 * Same rationale: the acquire_lock_block(L); <critical>; release_lock_block(L);
 * idiom deadlocks on lock-step SIMT GPUs without ITS (C500/warp64) when same-warp
 * lanes contend for the same block-scope lock. See with_lock() for details.
 */
template <typename i_t, typename critical_fn_t>
DI void with_lock_block(i_t* lock, critical_fn_t&& critical_section)
{
  bool done = false;
  while (!done) {
    if (atomicCAS_block(lock, 0, 1) == 0) {
      __threadfence_block();
      critical_section();
      __threadfence_block();
      atomicExch_block(lock, 0);
      done = true;
    }
  }
}

template <typename T>
DI void init_shmem(T& shmem, T val)
{
  if (threadIdx.x == 0) { shmem = val; }
}

template <typename T>
DI void init_block_shmem(T* shmem, T val, size_t size)
{
  for (auto i = threadIdx.x; i < size; i += blockDim.x) {
    shmem[i] = val;
  }
}

template <typename T>
DI void init_block_shmem(raft::device_span<T> sh_span, T val)
{
  init_block_shmem(sh_span.data(), val, sh_span.size());
}

template <typename T>
DI void block_sequence(T* arr, const size_t size)
{
  for (auto i = threadIdx.x; i < size; i += blockDim.x) {
    arr[i] = i;
  }
}

template <typename T>
DI void block_copy(T* dst, const T* src, const size_t size)
{
  for (auto i = threadIdx.x; i < size; i += blockDim.x) {
    dst[i] = src[i];
  }
}

template <typename T>
DI void block_copy(raft::device_span<T> dst,
                   const raft::device_span<const T> src,
                   const size_t size)
{
  cuopt_assert(src.size() >= size, "block_copy::src does not have the sufficient size");
  cuopt_assert(dst.size() >= size, "block_copy::dst does not have the sufficient size");
  block_copy(dst.data(), src.data(), size);
}

template <typename T>
DI void block_copy(raft::device_span<T> dst, const raft::device_span<T> src, const size_t size)
{
  cuopt_assert(src.size() >= size, "block_copy::src does not have the sufficient size");
  cuopt_assert(dst.size() >= size, "block_copy::dst does not have the sufficient size");
  block_copy(dst.data(), src.data(), size);
}

template <typename T>
DI void block_copy(raft::device_span<T> dst, const raft::device_span<T> src)
{
  cuopt_assert(dst.size() >= src.size(), "");
  block_copy(dst, src, src.size());
}

template <typename i_t>
i_t next_pow2(i_t val)
{
  return 1 << (raft::log2(val) + 1);
}

// FIXME:: handle alignment when dealing with different sized precisions
template <typename T, typename i_t>
static DI thrust::tuple<raft::device_span<T>, i_t*> wrap_ptr_as_span(i_t* shmem, size_t sz)
{
  T* sh_ptr = (T*)shmem;
  auto s    = raft::device_span<T>{sh_ptr, sz};

  sh_ptr = sh_ptr + sz;
  return thrust::make_tuple(s, (i_t*)sh_ptr);
}

template <class To, class From>
HDI To bit_cast(const From& src)
{
  static_assert(sizeof(To) == sizeof(From));
  return *(To*)(&src);
}

/**
 * @brief Raises the dynamic shared-memory limit for a CUDA kernel, with caching.
 *
 * Calls cudaFuncSetAttribute(cudaFuncAttributeMaxDynamicSharedMemorySize) only when
 * @p dynamic_request_size exceeds the previously set limit for @p function.  The
 * per-kernel high-water mark is stored in a process-wide cache so that repeated
 * calls with the same or smaller sizes are cheap shared-lock reads.
 *
 * Thread safety: safe to call concurrently from multiple host threads.
 *
 * @param function             Host pointer to the __global__ kernel function.
 * @param dynamic_request_size Requested dynamic shared memory in bytes.
 *                             A value of 0 is a no-op and always returns true.
 * @return true  if the attribute was successfully set (or was already sufficient).
 * @return false if cudaFuncSetAttribute failed (e.g. size exceeds device limit);
 *               the sticky CUDA error is consumed so it cannot surface later.
 */
template <typename Function>
inline bool set_shmem_of_kernel(Function* function, size_t dynamic_request_size)
{
  static std::shared_mutex mtx;
  static std::unordered_map<Function*, size_t> shmem_sizes;

  if (dynamic_request_size != 0) {
    dynamic_request_size = raft::alignTo(dynamic_request_size, size_t(1024));

    {
      std::shared_lock<std::shared_mutex> rlock(mtx);
      auto it = shmem_sizes.find(function);
      if (it != shmem_sizes.end() && dynamic_request_size <= it->second) { return true; }
    }

    std::unique_lock<std::shared_mutex> wlock(mtx);
    size_t current_size = shmem_sizes.count(function) ? shmem_sizes[function] : 0;
    if (dynamic_request_size > current_size) {
      auto err = cudaFuncSetAttribute(
        function, cudaFuncAttributeMaxDynamicSharedMemorySize, dynamic_request_size);
      if (err == cudaSuccess) {
        shmem_sizes[function] = dynamic_request_size;
        return true;
      } else {
        cudaGetLastError();  // clear sticky error so later RAFT_CHECK_CUDA doesn't catch it
        return false;
      }
    }
  }
  return true;
}

template <typename T>
DI void sorted_insert(T* array, T item, int curr_size, int max_size)
{
  for (int i = curr_size - 1; i >= 0; --i) {
    if (i == max_size - 1) continue;
    if (array[i] < item) {
      array[i + 1] = item;
      return;
    } else {
      array[i + 1] = array[i];
    }
  }
  array[0] = item;
}

inline size_t get_device_memory_size()
{
  size_t free_mem, total_mem;
  RAFT_CUDA_TRY(cudaMemGetInfo(&free_mem, &total_mem));
  // TODO (bdice): Restore limiting adaptor check after updating CCCL to support resource_cast
  return total_mem;
}

}  // namespace cuopt
