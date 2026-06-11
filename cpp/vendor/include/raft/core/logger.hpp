/*
 * cuOpt vendored RAFT shim — minimal logger surface (no-op).
 * cuOpt uses its own CUOPT_LOG_* macros; raft logging is unused.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <cstdio>

namespace raft {

enum class level_enum { trace, debug, info, warn, error, critical, off };

struct logger {
  static logger& get_global_logger()
  {
    static logger inst;
    return inst;
  }
  void set_level(level_enum) {}
  template <typename... Args>
  void log(level_enum, const char*, Args&&...)
  {
  }
};

inline logger& default_logger() { return logger::get_global_logger(); }

}  // namespace raft

#ifndef RAFT_LOG_INFO
#define RAFT_LOG_TRACE(...)    ((void)0)
#define RAFT_LOG_DEBUG(...)    ((void)0)
#define RAFT_LOG_INFO(...)     ((void)0)
#define RAFT_LOG_WARN(...)     ((void)0)
#define RAFT_LOG_ERROR(...)    ((void)0)
#define RAFT_LOG_CRITICAL(...) ((void)0)
#endif
