#include "compute_engine/platform/logger.h"

#include <spdlog/sinks/basic_file_sink.h>
#include <spdlog/sinks/stdout_color_sinks.h>
#include <spdlog/spdlog.h>

namespace compute_engine {

std::shared_ptr<spdlog::logger> LogManager::logger_ = nullptr;

void LogManager::Init(const std::string& log_file,
                      LogLevel level, bool color_console) {
    if (logger_) {
        return;
    }

    std::vector<spdlog::sink_ptr> sinks;

    if (color_console) {
        auto console_sink = std::make_shared<spdlog::sinks::stdout_color_sink_mt>();
        console_sink->set_pattern("[%^%l%$] %s:%#: %v");
        sinks.push_back(std::move(console_sink));
    }

    if (!log_file.empty()) {
        auto file_sink = std::make_shared<spdlog::sinks::basic_file_sink_mt>(log_file, true);
        file_sink->set_pattern("[%l] %s:%#: %v");
        sinks.push_back(std::move(file_sink));
    }

    if (sinks.empty()) {
        auto console_sink = std::make_shared<spdlog::sinks::stdout_color_sink_mt>();
        console_sink->set_pattern("[%^%l%$] %s:%#: %v");
        sinks.push_back(std::move(console_sink));
    }

    auto combined_logger = std::make_shared<spdlog::logger>("compute_engine", sinks.begin(), sinks.end());
    spdlog::level::level_enum spd_level = spdlog::level::info;
    switch (level) {
        case LogLevel::TRACE: spd_level = spdlog::level::trace; break;
        case LogLevel::DEBUG: spd_level = spdlog::level::debug; break;
        case LogLevel::INFO:  spd_level = spdlog::level::info;  break;
        case LogLevel::WARN:  spd_level = spdlog::level::warn;  break;
        case LogLevel::ERROR: spd_level = spdlog::level::err;   break;
        case LogLevel::FATAL: spd_level = spdlog::level::critical; break;
    }
    combined_logger->set_level(spd_level);
    combined_logger->flush_on(spd_level);
    combined_logger->set_pattern("[%^%l%$] %s:%#: %v");
    spdlog::register_logger(combined_logger);
    logger_ = std::move(combined_logger);

    LOGI("Logger initialized (level={})", LogLevelToString(level));
}

void LogManager::Shutdown() {
    if (logger_) {
        LOGI("Logger shutting down");
        logger_.reset();
        spdlog::shutdown();
    }
}

void LogManager::SetLevel(LogLevel level) {
    if (!logger_) return;
    spdlog::level::level_enum spd_level = spdlog::level::info;
    switch (level) {
        case LogLevel::TRACE: spd_level = spdlog::level::trace; break;
        case LogLevel::DEBUG: spd_level = spdlog::level::debug; break;
        case LogLevel::INFO:  spd_level = spdlog::level::info;  break;
        case LogLevel::WARN:  spd_level = spdlog::level::warn;  break;
        case LogLevel::ERROR: spd_level = spdlog::level::err;   break;
        case LogLevel::FATAL: spd_level = spdlog::level::critical; break;
    }
    logger_->set_level(spd_level);
    logger_->flush_on(spd_level);
    LOGI("Log level changed to {}", LogLevelToString(level));
}

void LogManager::SetLogFile(const std::string& log_file) {
    if (!logger_) return;
    auto file_sink = std::make_shared<spdlog::sinks::basic_file_sink_mt>(log_file, true);
    file_sink->set_pattern("[%l] %s:%#: %v");
    logger_->sinks().push_back(std::move(file_sink));
    LOGI("Log file set to {}", log_file);
}

std::shared_ptr<spdlog::logger> LogManager::GetLogger() {
    return logger_;
}

} // namespace compute_engine
