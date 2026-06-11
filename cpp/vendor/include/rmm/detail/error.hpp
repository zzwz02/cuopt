/*
 * cuOpt vendored RMM shim — replaces the external RAPIDS rmm dependency.
 * cuda::mr-free; allocations are backed by CUDA stream-ordered memory pools.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <cuda_runtime_api.h>

#include <cassert>
#include <stdexcept>
#include <string>

namespace rmm {

/** Thrown by RMM_EXPECTS / RMM_FAIL for logic errors. */
struct logic_error : public std::logic_error {
  using std::logic_error::logic_error;
};

/** Thrown by RMM_CUDA_TRY on a CUDA runtime error. */
struct cuda_error : public std::runtime_error {
  using std::runtime_error::runtime_error;
};

/** Base class for allocation failures. */
class bad_alloc : public std::bad_alloc {
 public:
  bad_alloc(const char* msg) : _what{std::string{std::bad_alloc::what()} + ": " + msg} {}
  bad_alloc(std::string const& msg) : bad_alloc{msg.c_str()} {}
  [[nodiscard]] const char* what() const noexcept override { return _what.c_str(); }

 private:
  std::string _what;
};

/** Thrown when the device memory resource cannot satisfy an allocation. */
class out_of_memory : public bad_alloc {
 public:
  out_of_memory(const char* msg) : bad_alloc{std::string{"out_of_memory: "} + msg} {}
  out_of_memory(std::string const& msg) : out_of_memory{msg.c_str()} {}
};

/** Thrown on out-of-bounds element access. */
struct out_of_range : public std::out_of_range {
  using std::out_of_range::out_of_range;
};

struct invalid_argument : public std::invalid_argument {
  using std::invalid_argument::invalid_argument;
};

}  // namespace rmm

#define RMM_STRINGIFY_DETAIL(x) #x
#define RMM_STRINGIFY(x)        RMM_STRINGIFY_DETAIL(x)
#define RMM_EXPAND(x)           x

/**
 * @brief Macro overload selector so RMM_EXPECTS accepts either
 * (cond, msg) -> rmm::logic_error, or (cond, msg, exception_type).
 */
#define RMM_GET_EXPECTS_MACRO(_1, _2, _3, NAME, ...) NAME

#define RMM_EXPECTS_2(cond, reason)                                       \
  do {                                                                    \
    if (!(cond)) { throw rmm::logic_error("RMM failure at: " __FILE__ ":" \
                     RMM_STRINGIFY(__LINE__) ": " reason); }              \
  } while (0)

#define RMM_EXPECTS_3(cond, reason, exception_type)                            \
  do {                                                                         \
    if (!(cond)) { throw exception_type{"RMM failure at: " __FILE__ ":"        \
                     RMM_STRINGIFY(__LINE__) ": " reason}; }                   \
  } while (0)

#define RMM_EXPECTS(...) \
  RMM_EXPAND(RMM_GET_EXPECTS_MACRO(__VA_ARGS__, RMM_EXPECTS_3, RMM_EXPECTS_2)(__VA_ARGS__))

#define RMM_FAIL_1(reason) \
  throw rmm::logic_error("RMM failure at: " __FILE__ ":" RMM_STRINGIFY(__LINE__) ": " reason)
#define RMM_FAIL_2(reason, exception_type) \
  throw exception_type { "RMM failure at: " __FILE__ ":" RMM_STRINGIFY(__LINE__) ": " reason }
#define RMM_GET_FAIL_MACRO(_1, _2, NAME, ...) NAME
#define RMM_FAIL(...) RMM_EXPAND(RMM_GET_FAIL_MACRO(__VA_ARGS__, RMM_FAIL_2, RMM_FAIL_1)(__VA_ARGS__))

/**
 * @brief Error-checking macro for CUDA runtime API calls. Throws rmm::cuda_error
 * (or a caller-specified exception) and clears the sticky error.
 */
#define RMM_CUDA_TRY(...)                                                    \
  RMM_EXPAND(RMM_GET_CUDA_TRY_MACRO(__VA_ARGS__, RMM_CUDA_TRY_2, RMM_CUDA_TRY_1)(__VA_ARGS__))
#define RMM_GET_CUDA_TRY_MACRO(_1, _2, NAME, ...) NAME

#define RMM_CUDA_TRY_1(call) RMM_CUDA_TRY_2(call, rmm::cuda_error)
#define RMM_CUDA_TRY_2(call, exception_type)                                            \
  do {                                                                                  \
    cudaError_t const status = (call);                                                  \
    if (cudaSuccess != status) {                                                        \
      cudaGetLastError();                                                               \
      throw exception_type{std::string{"CUDA error at: "} + __FILE__ + ":" +            \
                           RMM_STRINGIFY(__LINE__) + ": " + cudaGetErrorName(status) +  \
                           " " + cudaGetErrorString(status)};                           \
    }                                                                                   \
  } while (0)

/**
 * @brief Allocation-specific CUDA error check: maps cudaErrorMemoryAllocation to
 * rmm::out_of_memory, everything else to rmm::bad_alloc. Clears the sticky error.
 */
#define RMM_CUDA_TRY_ALLOC(...)                                                              \
  RMM_EXPAND(RMM_GET_CUDA_TRY_ALLOC_MACRO(__VA_ARGS__, RMM_CUDA_TRY_ALLOC_2,                 \
                                          RMM_CUDA_TRY_ALLOC_1)(__VA_ARGS__))
#define RMM_GET_CUDA_TRY_ALLOC_MACRO(_1, _2, NAME, ...) NAME
#define RMM_CUDA_TRY_ALLOC_1(call) RMM_CUDA_TRY_ALLOC_2(call, 0)
#define RMM_CUDA_TRY_ALLOC_2(call, bytes)                                                   \
  do {                                                                                      \
    cudaError_t const status = (call);                                                      \
    if (cudaSuccess != status) {                                                            \
      cudaGetLastError();                                                                   \
      auto const msg = std::string{"CUDA error at: "} + __FILE__ + ":" +                    \
                       RMM_STRINGIFY(__LINE__) + ": " + cudaGetErrorName(status) + " " +    \
                       cudaGetErrorString(status);                                          \
      if (status == cudaErrorMemoryAllocation) { throw rmm::out_of_memory{msg}; }           \
      throw rmm::bad_alloc{msg};                                                            \
    }                                                                                       \
  } while (0)

#ifndef RMM_ASSERT
#ifdef NDEBUG
#define RMM_ASSERT(...) ((void)0)
#else
#define RMM_ASSERT(cond, ...) assert((cond))
#endif
#endif

#ifndef RMM_LOGGING_ASSERT
#define RMM_LOGGING_ASSERT(cond) ((void)0)
#endif
