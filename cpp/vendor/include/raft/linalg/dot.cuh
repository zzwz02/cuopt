/*
 * cuOpt vendored RAFT shim — dot.cuh.
 * Re-exports the legacy pointer-form helpers and cuBLAS wrappers (cublasdot /
 * cublasnrm2) that the upstream header transitively provided.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <raft/linalg/binary_op.cuh>
#include <raft/linalg/detail/cublas_wrappers.hpp>
#include <raft/linalg/eltwise.cuh>
#include <raft/linalg/reduce.cuh>
#include <raft/linalg/unary_op.cuh>
