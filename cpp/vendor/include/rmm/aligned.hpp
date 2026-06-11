/*
 * cuOpt vendored RMM shim. SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <cstddef>
#include <cstdint>

namespace rmm {

/** Default alignment of CUDA device memory allocations (bytes). */
static constexpr std::size_t CUDA_ALLOCATION_ALIGNMENT{256};

[[nodiscard]] constexpr bool is_pow2(std::size_t value) noexcept
{
  return (value != 0U) && ((value & (value - 1)) == 0U);
}

[[nodiscard]] constexpr bool is_supported_alignment(std::size_t alignment) noexcept
{
  return is_pow2(alignment);
}

[[nodiscard]] constexpr std::size_t align_up(std::size_t value, std::size_t alignment) noexcept
{
  return (value + (alignment - 1)) & ~(alignment - 1);
}

[[nodiscard]] constexpr std::size_t align_down(std::size_t value, std::size_t alignment) noexcept
{
  return value & ~(alignment - 1);
}

[[nodiscard]] constexpr bool is_aligned(std::size_t value, std::size_t alignment) noexcept
{
  return value == align_down(value, alignment);
}

}  // namespace rmm
