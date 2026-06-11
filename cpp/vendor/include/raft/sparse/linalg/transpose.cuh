/*
 * cuOpt vendored RAFT shim — sparse/linalg/transpose.cuh.
 * raft::sparse::linalg::transpose is not called by cuOpt; this header also
 * re-exports the cuBLAS wrappers (cublassetpointermode) that consumers of the
 * upstream header relied on transitively.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <raft/core/handle.hpp>
#include <raft/linalg/detail/cublas_wrappers.hpp>
