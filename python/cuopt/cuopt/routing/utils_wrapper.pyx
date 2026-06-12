# SPDX-FileCopyrightText: Copyright (c) 2021-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved. # noqa
# SPDX-License-Identifier: Apache-2.0


# cython: profile=False
# distutils: language = c++
# cython: embedsignature = True
# cython: language_level = 3

from libc.stdint cimport uintptr_t

# handle_t / device_buffer come from routing_utilities' local shim
# declarations (RAPIDS-free; no pylibraft/rmm cimports).
from cuopt.routing.structure.routing_utilities cimport *

from enum import IntEnum

import cupy as cp
import numpy as np
from numba import cuda

import cudf

from libcpp.memory cimport unique_ptr
from libcpp.utility cimport move

# RAPIDS-free build: device buffers are copied to host numpy (then wrapped in
# cudf/cupy) instead of crossing the boundary as rmm DeviceBuffer, whose
# python class is compiled against the real rmm ABI.
cdef extern from "cuda_runtime_api.h" nogil:
    ctypedef enum cudaMemcpyKind:
        cudaMemcpyDeviceToHost
    int cudaMemcpy(void* dst, const void* src, size_t count,
                   cudaMemcpyKind kind)
    int cudaDeviceSynchronize()


cdef object _device_buffer_to_numpy(unique_ptr[device_buffer] buf, dtype):
    """Copy a device buffer to a host numpy array of the given dtype."""
    cdef device_buffer* b = buf.get()
    np_dtype = np.dtype(dtype)
    if b == NULL or b.size() == 0:
        return np.array([], dtype=np_dtype)
    cdef size_t nbytes = b.size()
    arr = np.empty(nbytes // np_dtype.itemsize, dtype=np_dtype)
    cdef uintptr_t dst = arr.__array_interface__['data'][0]
    cudaDeviceSynchronize()
    cudaMemcpy(<void*>dst, b.data(), nbytes, cudaMemcpyDeviceToHost)
    return arr


class DatasetDistribution(IntEnum):
    CLUSTERED = dataset_distribution_t.CLUSTERED
    RANDOM = dataset_distribution_t.RANDOM
    RANDOM_CLUSTERED = dataset_distribution_t.RANDOM_CLUSTERED


def generate_dataset(locations=100, asymmetric=True, min_demand=cudf.Series(),
                     max_demand=cudf.Series(), min_capacities=cudf.Series(),
                     max_capacities=cudf.Series(), min_service_time=0,
                     max_service_time=0, tw_tightness=0.0,
                     drop_return_trips=0.0, shifts=1,
                     n_vehicle_types=1, n_matrix_types=1,
                     distribution=DatasetDistribution.CLUSTERED,
                     center_box=None, seed=0):

    cdef unique_ptr[handle_t] handle_ptr
    handle_ptr.reset(new handle_t())
    handle_ = handle_ptr.get()

    min_demand = min_demand.astype(np.int16)
    max_demand = max_demand.astype(np.int16)
    min_capacities = min_capacities.astype(np.uint16)
    max_capacities = max_capacities.astype(np.uint16)
    dim = min_demand.shape[0]

    cdef uintptr_t c_min_demand = <uintptr_t>NULL
    if min_demand is not None:
        c_min_demand = min_demand.__cuda_array_interface__['data'][0]
    cdef uintptr_t c_max_demand = <uintptr_t>NULL
    if max_demand is not None:
        c_max_demand = max_demand.__cuda_array_interface__['data'][0]
    cdef uintptr_t c_min_capacities = <uintptr_t>NULL
    if min_capacities is not None:
        c_min_capacities = min_capacities.__cuda_array_interface__['data'][0]
    cdef uintptr_t c_max_capacities = <uintptr_t>NULL
    if max_capacities is not None:
        c_max_capacities = max_capacities.__cuda_array_interface__['data'][0]
    cdef int c_distrib_type = distribution.value

    center_box_min = 0
    center_box_max = locations / 2
    if center_box is not None:
        center_box_min = center_box[0]
        center_box_max = center_box[1]

    cdef dataset_params_t[int, float] params
    populate_dataset_params[int, float](params, locations, asymmetric,
                                        dim, <int32_t*>c_min_demand,
                                        <int32_t*>c_max_demand,
                                        <int32_t*>c_min_capacities,
                                        <int32_t*>c_max_capacities,
                                        min_service_time,
                                        max_service_time,
                                        tw_tightness,
                                        drop_return_trips,
                                        shifts,
                                        n_vehicle_types,
                                        n_matrix_types,
                                        <dataset_distribution_t>c_distrib_type,
                                        <float>center_box_min,
                                        <float>center_box_max,
                                        seed)

    g_ret_ptr = move(call_generate_dataset(handle_[0], params))
    g_ret = move(g_ret_ptr.get()[0])

    coordinates = cudf.DataFrame()
    orders = cudf.DataFrame()
    vehicles = cudf.DataFrame()
    constraints = dict()

    coordinates['x'] = cudf.Series(
        _device_buffer_to_numpy(move(g_ret.d_x_pos_), np.float32))
    coordinates['y'] = cudf.Series(
        _device_buffer_to_numpy(move(g_ret.d_y_pos_), np.float32))

    matrices_np = _device_buffer_to_numpy(move(g_ret.d_matrices_), np.float32)
    matrices_np = matrices_np.reshape(
        (n_vehicle_types, n_matrix_types, locations, locations))
    matrices_ret = cp.array(matrices_np)

    # Create vehicles_df
    vehicles["earliest_time"] = cudf.Series(
        _device_buffer_to_numpy(move(g_ret.d_vehicle_earliest_time_), np.int32))
    vehicles["latest_time"] = cudf.Series(
        _device_buffer_to_numpy(move(g_ret.d_vehicle_latest_time_), np.int32))
    vehicles["drop_return_trips"] = cudf.Series(
        _device_buffer_to_numpy(move(g_ret.d_drop_return_trips_), np.bool_))
    vehicles["skip_first_trips"] = cudf.Series(
        _device_buffer_to_numpy(move(g_ret.d_skip_first_trips_), np.bool_))

    fleet_size = vehicles["earliest_time"].shape[0]
    capacities_np = _device_buffer_to_numpy(move(g_ret.d_caps_), np.uint16)
    capacities = cp.array(capacities_np.reshape((dim, fleet_size)))
    for i in range(dim):
        vehicles["capacity_" + str(i)] = capacities[i]

    # Fleet order constraints
    service_times_np = _device_buffer_to_numpy(
        move(g_ret.d_service_time_), np.int32)
    order_service_times = cp.array(
        service_times_np.reshape((fleet_size, locations)))

    constraints["order_service_times"] = order_service_times

    # Create orders df
    orders["earliest_time"] = cudf.Series(
        _device_buffer_to_numpy(move(g_ret.d_earliest_time_), np.int32))
    orders["latest_time"] = cudf.Series(
        _device_buffer_to_numpy(move(g_ret.d_latest_time_), np.int32))

    demands_np = _device_buffer_to_numpy(move(g_ret.d_demands_), np.int16)
    demands = cp.array(demands_np.reshape((dim, locations)))
    for i in range(dim):
        orders["demand_" + str(i)] = demands[i]

    return coordinates, matrices_ret, orders, vehicles, constraints
