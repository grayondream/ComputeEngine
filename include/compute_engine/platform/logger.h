#pragma once

#include "compute_engine/core/macros.h"

#include <memory>
#include <string>

namespace spdlog {
class logger;
}

namespace compute_engine {

enum class LogLevel {
    TRACE = 0,
    DEBUG = 1,
    INFO  = 2,
    WARN  = 3,
    ERROR = 4,
    FATAL = 5,
};

class LogManager {
public:
    static void Init(const std::string& log_file = "",
                     LogLevel level = LogLevel::INFO,
                     bool color_console = true);
    static void Shutdown();
    static void SetLevel(LogLevel level);
    static void SetLogFile(const std::string& log_file);
    static std::shared_ptr<spdlog::logger> GetLogger();

private:
    static std::shared_ptr<spdlog::logger> logger_;
    LogManager() = delete;
};

inline std::string LogLevelToString(LogLevel level) {
    switch (level) {
        case LogLevel::TRACE: return "TRACE";
        case LogLevel::DEBUG: return "DEBUG";
        case LogLevel::INFO:  return "INFO";
        case LogLevel::WARN:  return "WARN";
        case LogLevel::ERROR: return "ERROR";
        case LogLevel::FATAL: return "FATAL";
    }
    return "UNKNOWN";
}

} // namespace compute_engine

#ifdef CE_ENABLE_LOGGING

#include <spdlog/spdlog.h>

#define LOGT(...) do { auto* _lg = ::compute_engine::LogManager::GetLogger().get(); if (_lg) SPDLOG_LOGGER_CALL(_lg, spdlog::level::trace, __VA_ARGS__); } while(0)
#define LOGD(...) do { auto* _lg = ::compute_engine::LogManager::GetLogger().get(); if (_lg) SPDLOG_LOGGER_CALL(_lg, spdlog::level::debug, __VA_ARGS__); } while(0)
#define LOGI(...) do { auto* _lg = ::compute_engine::LogManager::GetLogger().get(); if (_lg) SPDLOG_LOGGER_CALL(_lg, spdlog::level::info, __VA_ARGS__); } while(0)
#define LOGW(...) do { auto* _lg = ::compute_engine::LogManager::GetLogger().get(); if (_lg) SPDLOG_LOGGER_CALL(_lg, spdlog::level::warn, __VA_ARGS__); } while(0)
#define LOGE(...) do { auto* _lg = ::compute_engine::LogManager::GetLogger().get(); if (_lg) SPDLOG_LOGGER_CALL(_lg, spdlog::level::err, __VA_ARGS__); } while(0)
#define LOGF(...) do { auto* _lg = ::compute_engine::LogManager::GetLogger().get(); if (_lg) SPDLOG_LOGGER_CALL(_lg, spdlog::level::critical, __VA_ARGS__); } while(0)

#define CE_LOG_TRACE(...) LOGT(__VA_ARGS__)
#define CE_LOG_DEBUG(...) LOGD(__VA_ARGS__)
#define CE_LOG_INFO(...)  LOGI(__VA_ARGS__)
#define CE_LOG_WARN(...)  LOGW(__VA_ARGS__)
#define CE_LOG_ERROR(...) LOGE(__VA_ARGS__)
#define CE_LOG_FATAL(...) LOGF(__VA_ARGS__)

#else

#define LOGT(...) ((void)0)
#define LOGD(...) ((void)0)
#define LOGI(...) ((void)0)
#define LOGW(...) ((void)0)
#define LOGE(...) ((void)0)
#define LOGF(...) ((void)0)

#define CE_LOG_TRACE(...) ((void)0)
#define CE_LOG_DEBUG(...) ((void)0)
#define CE_LOG_INFO(...)  ((void)0)
#define CE_LOG_WARN(...)  ((void)0)
#define CE_LOG_ERROR(...) ((void)0)
#define CE_LOG_FATAL(...) ((void)0)

#endif
