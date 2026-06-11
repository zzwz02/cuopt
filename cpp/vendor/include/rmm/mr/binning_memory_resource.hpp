/*
 * cuOpt vendored RMM shim — binning resource (delegates to its upstream pool).
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <rmm/cuda_stream_view.hpp>
#include <rmm/mr/device_memory_resource.hpp>

#include <cstddef>

namespace rmm::mr {

template <typename Upstream>
class binning_memory_resource final : public device_memory_resource {
 public:
  explicit binning_memory_resource(Upstream* upstream) : upstream_{upstream} {}
  binning_memory_resource(Upstream* upstream, int8_t, int8_t) : upstream_{upstream} {}
  binning_memory_resource(Upstream& upstream) : upstream_{&upstream} {}
  binning_memory_resource(Upstream& upstream, int8_t, int8_t) : upstream_{&upstream} {}

 private:
  void* do_allocate(std::size_t bytes, cuda_stream_view stream) override
  {
    return upstream_->allocate(bytes, stream);
  }
  void do_deallocate(void* ptr, std::size_t bytes, cuda_stream_view stream) override
  {
    upstream_->deallocate(ptr, bytes, stream);
  }
  [[nodiscard]] bool do_is_equal(device_memory_resource const& other) const noexcept override
  {
    return this == &other;
  }

  Upstream* upstream_;
};

template <typename Upstream>
binning_memory_resource(Upstream&) -> binning_memory_resource<Upstream>;
template <typename Upstream>
binning_memory_resource(Upstream&, int8_t, int8_t) -> binning_memory_resource<Upstream>;

}  // namespace rmm::mr
