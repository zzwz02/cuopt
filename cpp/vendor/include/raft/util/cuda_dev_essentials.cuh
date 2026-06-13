/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <raft/core/detail/macros.hpp>
#include <raft/util/warp_constants.hpp>

#include <cuda_fp16.h>

// This file provides a few essential functions for use in __device__ code. The
// scope is necessarily limited to ensure that compilation times are minimized.
// Please make sure not to include large / expensive files from here.

namespace RAFT_EXPORT raft {

/** helper macro for device inlined functions */
#define DI  inline __device__
#define HDI inline __host__ __device__
#define HD  __host__ __device__

/**
 * @brief Provide a ceiling division operation ie. ceil(a / b)
 * @tparam IntType supposed to be only integers for now!
 */
template <typename IntType>
constexpr HDI IntType ceildiv(IntType a, IntType b)
{
  return (a + b - 1) / b;
}

/**
 * @brief Provide an alignment function ie. ceil(a / b) * b
 * @tparam IntType supposed to be only integers for now!
 */
template <typename IntType>
constexpr HDI IntType alignTo(IntType a, IntType b)
{
  return ceildiv(a, b) * b;
}

/**
 * @brief Provide an alignment function ie. (a / b) * b
 * @tparam IntType supposed to be only integers for now!
 */
template <typename IntType>
constexpr HDI IntType alignDown(IntType a, IntType b)
{
  return (a / b) * b;
}

/**
 * @brief Check if the input is a power of 2
 * @tparam IntType data type (checked only for integers)
 */
template <typename IntType>
constexpr HDI bool isPo2(IntType num)
{
  return (num && !(num & (num - 1)));
}

/**
 * @brief Give logarithm of the number to base-2
 * @tparam IntType data type (checked only for integers)
 */
template <typename IntType>
constexpr HDI IntType log2(IntType num, IntType ret = IntType(0))
{
  return num <= IntType(1) ? ret : log2(num >> IntType(1), ++ret);
}

/** get the laneId of the current thread */
DI int laneId()
{
#if defined(__HIP_DEVICE_COMPILE__) || defined(__MACA_ARCH__)
  return __lane_id();
#else
  int id;
  asm("mov.s32 %0, %%laneid;" : "=r"(id));
  return id;
#endif
}

/**
 * @brief Number of set bits in a lane mask (ballot/activemask result).
 * Compile-time dispatch on the platform's lane-mask width.
 */
DI int lane_popc(lane_mask_t mask)
{
  if constexpr (sizeof(lane_mask_t) == 8) {
    return __popcll(static_cast<unsigned long long>(mask));
  } else {
    return __popc(static_cast<unsigned int>(mask));
  }
}

/** @brief 1-based index of the lowest set lane bit; 0 when the mask is empty. */
DI int lane_ffs(lane_mask_t mask)
{
  if constexpr (sizeof(lane_mask_t) == 8) {
    return __ffsll(static_cast<long long>(mask));
  } else {
    return __ffs(static_cast<int>(mask));
  }
}

/** @brief Index of the highest set lane bit; -1 when the mask is empty. */
DI int lane_fls(lane_mask_t mask)
{
  if constexpr (sizeof(lane_mask_t) == 8) {
    return mask == 0 ? -1 : 63 - __clzll(static_cast<long long>(mask));
  } else {
    return mask == 0 ? -1 : 31 - __clz(static_cast<int>(mask));
  }
}

/** @brief Mask of currently-active lanes, in the platform's lane-mask width. */
DI lane_mask_t activemask()
{
#if defined(__HIP_DEVICE_COMPILE__)
  return __ballot(1);
#else
  return __activemask();
#endif
}

/** Device function to apply the input lambda across threads in the grid */
template <int ItemsPerThread, typename L>
DI void forEach(int num, L lambda)
{
  int idx              = (blockDim.x * blockIdx.x) + threadIdx.x;
  const int numThreads = blockDim.x * gridDim.x;
#pragma unroll
  for (int itr = 0; itr < ItemsPerThread; ++itr, idx += numThreads) {
    if (idx < num) lambda(idx, itr);
  }
}

/**
 * @brief Swap two values
 * @tparam T the datatype of the values
 * @param a first input
 * @param b second input
 */
template <typename T>
HDI void swapVals(T& a, T& b)
{
  T tmp = a;
  a     = b;
  b     = tmp;
}

/**
 * @brief Convert half to float
 * @tparam T the datatype of the value
 * @param a need to convert
 */
template <typename T>
HDI auto to_float(T& a)
{
  if constexpr (std::is_same_v<typename std::remove_const<T>::type, half>) {
    return __half2float(a);
  } else {
    return a;
  }
}

}  // namespace RAFT_EXPORT raft
