/*
 * cuOpt vendored RAFT shim — divide.cuh.
 * Included by cuOpt; re-exports the legacy pointer-form helpers (reduce /
 * eltwise / binaryOp) that the upstream header transitively provided.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <raft/linalg/binary_op.cuh>
#include <raft/linalg/eltwise.cuh>
#include <raft/linalg/reduce.cuh>
#include <raft/linalg/unary_op.cuh>
