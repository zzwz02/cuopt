/*
 * cuOpt vendored rapids_logger shim — header-only, printf-style logger.
 * Replaces the compiled librapids_logger.so. Matches the subset of the
 * rapids_logger API used by cuOpt (logger, level_enum, ostream/file/callback
 * sinks). SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <rapids_logger/log_levels.h>

#include <cstdint>
#include <cstdio>
#include <ctime>
#include <fstream>
#include <functional>
#include <iostream>
#include <memory>
#include <mutex>
#include <string>
#include <utility>
#include <vector>

namespace rapids_logger {

enum class level_enum : int32_t {
  trace    = RAPIDS_LOGGER_LOG_LEVEL_TRACE,
  debug    = RAPIDS_LOGGER_LOG_LEVEL_DEBUG,
  info     = RAPIDS_LOGGER_LOG_LEVEL_INFO,
  warn     = RAPIDS_LOGGER_LOG_LEVEL_WARN,
  error    = RAPIDS_LOGGER_LOG_LEVEL_ERROR,
  critical = RAPIDS_LOGGER_LOG_LEVEL_CRITICAL,
  off      = RAPIDS_LOGGER_LOG_LEVEL_OFF,
};

inline bool operator<(level_enum a, level_enum b) { return int32_t(a) < int32_t(b); }
inline bool operator<=(level_enum a, level_enum b) { return int32_t(a) <= int32_t(b); }
inline bool operator>(level_enum a, level_enum b) { return int32_t(a) > int32_t(b); }
inline bool operator>=(level_enum a, level_enum b) { return int32_t(a) >= int32_t(b); }

/** @brief Base class for log sinks. */
class sink {
 public:
  virtual ~sink()                                                    = default;
  virtual void log(level_enum lvl, std::string const& formatted_msg) = 0;
  virtual void flush()                                               = 0;
};

using sink_ptr        = std::shared_ptr<sink>;
using log_callback_t  = std::function<void(int lvl, const char* msg)>;
using flush_callback_t = std::function<void()>;

/** @brief Sink that writes formatted messages to a std::ostream. */
class ostream_sink_mt : public sink {
 public:
  explicit ostream_sink_mt(std::ostream& stream, bool force_flush = false)
    : stream_{stream}, force_flush_{force_flush}
  {
  }
  void log(level_enum, std::string const& msg) override
  {
    std::lock_guard<std::mutex> lock{mtx_};
    stream_ << msg << '\n';
    if (force_flush_) { stream_.flush(); }
  }
  void flush() override
  {
    std::lock_guard<std::mutex> lock{mtx_};
    stream_.flush();
  }

 private:
  std::ostream& stream_;
  bool force_flush_;
  std::mutex mtx_;
};

/** @brief Sink that appends/truncates a log file. */
class basic_file_sink_mt : public sink {
 public:
  explicit basic_file_sink_mt(std::string const& filename, bool truncate = false)
    : file_{filename, truncate ? std::ios::trunc : std::ios::app}
  {
  }
  void log(level_enum, std::string const& msg) override
  {
    std::lock_guard<std::mutex> lock{mtx_};
    file_ << msg << '\n';
  }
  void flush() override
  {
    std::lock_guard<std::mutex> lock{mtx_};
    file_.flush();
  }

 private:
  std::ofstream file_;
  std::mutex mtx_;
};

/** @brief Sink that forwards messages to a user callback. */
class callback_sink_mt : public sink {
 public:
  explicit callback_sink_mt(log_callback_t const& callback,
                            flush_callback_t const& flush = nullptr)
    : callback_{callback}, flush_{flush}
  {
  }
  void log(level_enum lvl, std::string const& msg) override
  {
    if (callback_) { callback_(static_cast<int>(lvl), msg.c_str()); }
  }
  void flush() override
  {
    if (flush_) { flush_(); }
  }

 private:
  log_callback_t callback_;
  flush_callback_t flush_;
};

/** @brief Minimal printf-style logger fanning out to a set of sinks. */
class logger {
 public:
  logger()                         = delete;
  logger(logger const&)            = delete;
  logger& operator=(logger const&) = delete;

  // Movable despite the std::mutex member (a moved-into logger gets a fresh
  // mutex; the sink_vector holds no back-reference to the parent).
  logger(logger&& other) noexcept
    : name_{std::move(other.name_)},
      sinks_{std::move(other.sinks_)},
      level_{other.level_},
      flush_level_{other.flush_level_},
      pattern_{std::move(other.pattern_)}
  {
  }

  logger(std::string name, std::vector<sink_ptr> sinks)
    : name_{std::move(name)}, sinks_{std::move(sinks)}
  {
  }
  logger(std::string name, std::string filename)
    : logger{std::move(name),
             std::vector<sink_ptr>{std::make_shared<basic_file_sink_mt>(filename, true)}}
  {
  }
  logger(std::string name, std::ostream& stream)
    : logger{std::move(name), std::vector<sink_ptr>{std::make_shared<ostream_sink_mt>(stream)}}
  {
  }
  ~logger() = default;

  /** @brief Container of sinks with the subset of vector ops cuOpt uses. */
  class sink_vector {
   public:
    using Iterator      = std::vector<sink_ptr>::iterator;
    using ConstIterator = std::vector<sink_ptr>::const_iterator;

    sink_vector() = default;
    explicit sink_vector(std::vector<sink_ptr> sinks) : sinks_{std::move(sinks)} {}
    void push_back(sink_ptr const& s) { sinks_.push_back(s); }
    void push_back(sink_ptr&& s) { sinks_.push_back(std::move(s)); }
    void pop_back() { sinks_.pop_back(); }
    void clear() { sinks_.clear(); }
    [[nodiscard]] Iterator begin() { return sinks_.begin(); }
    [[nodiscard]] Iterator end() { return sinks_.end(); }
    [[nodiscard]] ConstIterator begin() const { return sinks_.begin(); }
    [[nodiscard]] ConstIterator end() const { return sinks_.end(); }
    [[nodiscard]] std::size_t size() const { return sinks_.size(); }

   private:
    std::vector<sink_ptr> sinks_;
  };

  void log(level_enum lvl, std::string const& message)
  {
    if (!should_log(lvl)) { return; }
    std::string const out = format_message(lvl, message);
    std::lock_guard<std::mutex> lock{mtx_};
    for (auto& s : sinks_) {
      s->log(lvl, out);
      if (lvl >= flush_level_) { s->flush(); }
    }
  }

  // Named-level conveniences, matching rapids_logger's API (used e.g. by the
  // cuOpt gRPC server logger; the core cuOpt logger goes through macros).
  template <typename... Args>
  void trace(std::string const& format, Args&&... args)
  {
    log(level_enum::trace, format, std::forward<Args>(args)...);
  }
  template <typename... Args>
  void debug(std::string const& format, Args&&... args)
  {
    log(level_enum::debug, format, std::forward<Args>(args)...);
  }
  template <typename... Args>
  void info(std::string const& format, Args&&... args)
  {
    log(level_enum::info, format, std::forward<Args>(args)...);
  }
  template <typename... Args>
  void warn(std::string const& format, Args&&... args)
  {
    log(level_enum::warn, format, std::forward<Args>(args)...);
  }
  template <typename... Args>
  void error(std::string const& format, Args&&... args)
  {
    log(level_enum::error, format, std::forward<Args>(args)...);
  }
  template <typename... Args>
  void critical(std::string const& format, Args&&... args)
  {
    log(level_enum::critical, format, std::forward<Args>(args)...);
  }

  template <typename... Args>
  void log(level_enum lvl, std::string const& format, Args&&... args)
  {
    if (!should_log(lvl)) { return; }
    if (sizeof...(Args) == 0) {
      log(lvl, std::string{format});
      return;
    }
    auto to_c = [](auto&& arg) -> decltype(auto) {
      using ArgType = std::decay_t<decltype(arg)>;
      if constexpr (std::is_same_v<ArgType, std::string>) {
        return arg.c_str();
      } else {
        return std::forward<decltype(arg)>(arg);
      }
    };
    int const sz = std::snprintf(nullptr, 0, format.c_str(), to_c(std::forward<Args>(args))...);
    if (sz < 0) { return; }
    std::string buf(static_cast<std::size_t>(sz) + 1, '\0');
    std::snprintf(buf.data(), buf.size(), format.c_str(), to_c(std::forward<Args>(args))...);
    buf.resize(static_cast<std::size_t>(sz));
    log(lvl, buf);
  }

  void flush()
  {
    std::lock_guard<std::mutex> lock{mtx_};
    for (auto& s : sinks_) { s->flush(); }
  }

  [[nodiscard]] const sink_vector& sinks() const { return sinks_; }
  [[nodiscard]] sink_vector& sinks() { return sinks_; }

  [[nodiscard]] level_enum level() const { return level_; }
  void set_level(level_enum log_level) { level_ = log_level; }
  void flush_on(level_enum log_level) { flush_level_ = log_level; }
  [[nodiscard]] level_enum flush_level() const { return flush_level_; }
  [[nodiscard]] bool should_log(level_enum msg_level) const
  {
    return msg_level >= level_ && level_ != level_enum::off;
  }
  void set_pattern(std::string pattern) { pattern_ = std::move(pattern); }

 private:
  std::string format_message(level_enum, std::string const& message)
  {
    if (pattern_ == "%v") { return message; }
    // Non-trivial pattern: prefix a timestamp (sufficient for cuOpt's needs).
    std::time_t t = std::time(nullptr);
    char ts[32]{};
    std::tm tm_buf{};
#if defined(_WIN32)
    localtime_s(&tm_buf, &t);
#else
    localtime_r(&t, &tm_buf);
#endif
    std::strftime(ts, sizeof(ts), "[%Y-%m-%d %H:%M:%S] ", &tm_buf);
    return std::string{ts} + message;
  }

  std::string name_;
  sink_vector sinks_;
  level_enum level_{level_enum::info};
  level_enum flush_level_{level_enum::off};
  std::string pattern_{"%v"};
  std::mutex mtx_;
};

}  // namespace rapids_logger
