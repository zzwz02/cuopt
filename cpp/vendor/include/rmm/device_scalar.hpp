/*
 * cuOpt vendored RMM shim — single-element device allocation.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <rmm/cuda_stream_view.hpp>
#include <rmm/detail/error.hpp>
#include <rmm/device_uvector.hpp>
#include <rmm/mr/per_device_resource.hpp>
#include <rmm/resource_ref.hpp>

#include <cstddef>

namespace rmm {

template <typename T>
class device_scalar {
  static_assert(std::is_trivially_copyable_v<T>,
                "device_scalar only supports types that are trivially copyable.");

 public:
  using value_type    = typename device_uvector<T>::value_type;
  using reference      = typename device_uvector<T>::reference;
  using const_reference = typename device_uvector<T>::const_reference;
  using pointer        = typename device_uvector<T>::pointer;
  using const_pointer  = typename device_uvector<T>::const_pointer;

  ~device_scalar()                                = default;
  device_scalar(device_scalar&&) noexcept         = default;
  device_scalar& operator=(device_scalar&&) noexcept = default;
  device_scalar(device_scalar const&)             = delete;
  device_scalar& operator=(device_scalar const&)  = delete;
  device_scalar()                                 = delete;

  explicit device_scalar(cuda_stream_view stream,
                         device_async_resource_ref mr = mr::get_current_device_resource_ref())
    : storage_{1, stream, mr}
  {
  }

  explicit device_scalar(value_type const& initial_value,
                         cuda_stream_view stream,
                         device_async_resource_ref mr = mr::get_current_device_resource_ref())
    : storage_{1, stream, mr}
  {
    set_value_async(initial_value, stream);
  }

  device_scalar(device_scalar const& other,
                cuda_stream_view stream,
                device_async_resource_ref mr = mr::get_current_device_resource_ref())
    : storage_{other.storage_, stream, mr}
  {
  }

  [[nodiscard]] value_type value(cuda_stream_view stream) const { return storage_.front_element(stream); }

  void set_value_async(value_type const& value, cuda_stream_view stream)
  {
    storage_.set_element_async(0, value, stream);
  }
  void set_value_async(value_type&&, cuda_stream_view) = delete;

  void set_value_to_zero_async(cuda_stream_view stream)
  {
    storage_.set_element_to_zero_async(0, stream);
  }

  [[nodiscard]] pointer data() noexcept { return storage_.data(); }
  [[nodiscard]] const_pointer data() const noexcept { return storage_.data(); }

  [[nodiscard]] cuda_stream_view stream() const noexcept { return storage_.stream(); }
  void set_stream(cuda_stream_view stream) noexcept { storage_.set_stream(stream); }

 private:
  device_uvector<T> storage_;
};

}  // namespace rmm
