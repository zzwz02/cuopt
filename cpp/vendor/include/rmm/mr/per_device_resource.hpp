/*
 * cuOpt vendored RMM shim — current/per-device resource registry (cuda::mr-free).
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <rmm/cuda_device.hpp>
#include <rmm/mr/cuda_async_memory_resource.hpp>
#include <rmm/mr/device_memory_resource.hpp>
#include <rmm/resource_ref.hpp>

#include <array>
#include <mutex>
#include <type_traits>

namespace rmm::mr {

template <typename Resource>
inline constexpr bool is_device_mr_v =
  std::is_base_of_v<device_memory_resource, std::remove_reference_t<Resource>>;

namespace detail {

inline constexpr int max_devices = 256;

inline std::mutex& map_mutex()
{
  static std::mutex mtx;
  return mtx;
}

inline std::array<device_memory_resource*, max_devices>& resource_table()
{
  static std::array<device_memory_resource*, max_devices> table{};
  return table;
}

// Lazily-created default resource (stream-ordered pool) for a device. The whole
// codebase is written against stream-ordered semantics, so the default matches.
inline device_memory_resource* initial_resource(cuda_device_id id)
{
  static std::array<cuda_async_memory_resource*, max_devices> defaults{};
  cuda_set_device_raii set_dev{id};
  if (defaults[id.value()] == nullptr) {
    defaults[id.value()] = new cuda_async_memory_resource{};
  }
  return defaults[id.value()];
}

}  // namespace detail

inline device_memory_resource* get_per_device_resource(cuda_device_id id)
{
  std::lock_guard<std::mutex> lock{detail::map_mutex()};
  auto& slot = detail::resource_table()[id.value()];
  if (slot == nullptr) { slot = detail::initial_resource(id); }
  return slot;
}

inline device_memory_resource* set_per_device_resource(cuda_device_id id,
                                                       device_memory_resource* new_mr)
{
  std::lock_guard<std::mutex> lock{detail::map_mutex()};
  auto& slot    = detail::resource_table()[id.value()];
  auto* old_mr  = (slot == nullptr) ? detail::initial_resource(id) : slot;
  slot          = (new_mr == nullptr) ? detail::initial_resource(id) : new_mr;
  return old_mr;
}

// Accept a concrete resource by reference (CCCL-MR-style call sites pass a value/ref).
template <typename Resource, typename = std::enable_if_t<is_device_mr_v<Resource>>>
inline device_memory_resource* set_per_device_resource(cuda_device_id id, Resource& mr)
{
  return set_per_device_resource(id, static_cast<device_memory_resource*>(&mr));
}

inline device_memory_resource* get_current_device_resource()
{
  return get_per_device_resource(get_current_cuda_device());
}

inline device_memory_resource* set_current_device_resource(device_memory_resource* new_mr)
{
  return set_per_device_resource(get_current_cuda_device(), new_mr);
}

template <typename Resource, typename = std::enable_if_t<is_device_mr_v<Resource>>>
inline device_memory_resource* set_current_device_resource(Resource& mr)
{
  return set_current_device_resource(static_cast<device_memory_resource*>(&mr));
}

inline device_async_resource_ref get_current_device_resource_ref()
{
  return device_async_resource_ref{get_current_device_resource()};
}

inline device_async_resource_ref get_per_device_resource_ref(cuda_device_id id)
{
  return device_async_resource_ref{get_per_device_resource(id)};
}

inline device_async_resource_ref set_current_device_resource_ref(device_async_resource_ref next)
{
  return device_async_resource_ref{set_current_device_resource(next.get())};
}

inline device_async_resource_ref set_per_device_resource_ref(cuda_device_id id,
                                                             device_async_resource_ref next)
{
  return device_async_resource_ref{set_per_device_resource(id, next.get())};
}

}  // namespace rmm::mr
