/*
 * cuOpt vendored RMM shim — pool resource.
 *
 * The real stream-ordered pooling is provided by the CUDA driver mempool used
 * in cuda_async_memory_resource; this class is a thin compatibility wrapper so
 * existing `pool_memory_resource<Upstream>(upstream, size)` call sites compile.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <rmm/cuda_stream_view.hpp>
#include <rmm/mr/cuda_async_memory_resource.hpp>
#include <rmm/mr/device_memory_resource.hpp>

#include <cstddef>
#include <optional>

namespace rmm::mr {

template <typename Upstream = device_memory_resource>
class pool_memory_resource final : public device_memory_resource {
 public:
  explicit pool_memory_resource(Upstream*,
                                std::optional<std::size_t> initial_pool_size = {},
                                std::optional<std::size_t> maximum_pool_size = {})
    : impl_{initial_pool_size, maximum_pool_size}
  {
  }

  explicit pool_memory_resource(Upstream&,
                                std::optional<std::size_t> initial_pool_size = {},
                                std::optional<std::size_t> maximum_pool_size = {})
    : impl_{initial_pool_size, maximum_pool_size}
  {
  }

  explicit pool_memory_resource(Upstream&&,
                                std::optional<std::size_t> initial_pool_size = {},
                                std::optional<std::size_t> maximum_pool_size = {})
    : impl_{initial_pool_size, maximum_pool_size}
  {
  }

  [[nodiscard]] cudaMemPool_t pool_handle() const noexcept { return impl_.pool_handle(); }

 private:
  void* do_allocate(std::size_t bytes, cuda_stream_view stream) override
  {
    return impl_.allocate(bytes, stream);
  }
  void do_deallocate(void* ptr, std::size_t bytes, cuda_stream_view stream) override
  {
    impl_.deallocate(ptr, bytes, stream);
  }
  [[nodiscard]] bool do_is_equal(device_memory_resource const& other) const noexcept override
  {
    return this == &other;
  }

  cuda_async_memory_resource impl_;
};

// Deduction guide for `pool_memory_resource(make_async(), size)` style construction.
template <typename Upstream>
pool_memory_resource(Upstream&, std::optional<std::size_t> = {}, std::optional<std::size_t> = {})
  -> pool_memory_resource<Upstream>;
template <typename Upstream>
pool_memory_resource(Upstream&&, std::optional<std::size_t> = {}, std::optional<std::size_t> = {})
  -> pool_memory_resource<Upstream>;

}  // namespace rmm::mr
