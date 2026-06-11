/*
 * cuOpt vendored shim — common std headers.
 *
 * The upstream rmm/raft headers transitively pulled in many <map>, <vector>,
 * <tuple>, ... includes that some cuOpt translation units relied on without
 * including them directly. The leaner shim does not, so this header restores
 * that transitive surface from the foundational rmm chokepoint header.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <map>
#include <memory>
#include <optional>
#include <set>
#include <string>
#include <tuple>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>
