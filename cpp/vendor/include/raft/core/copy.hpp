/*
 * cuOpt vendored RAFT shim — raft::copy (pointer + length form).
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <raft/util/cudart_utils.hpp>

// raft::copy(dst, src, len, stream) is declared in cudart_utils.hpp.
