/*
 * cuOpt vendored RAFT shim — norm.cuh.
 * Header is included by cuOpt but its functions are not called; the legacy
 * pointer-form helpers cuOpt uses live in eltwise.cuh / binary_op.cuh / reduce.cuh.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <raft/util/cudart_utils.hpp>
