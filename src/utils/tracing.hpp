#pragma once

#include <string>
#include <iostream>
#include <sstream>
#include <chrono>
#include <iomanip>

namespace duckdb {
namespace anofox_statistics {

/**
 * @brief Configurable tracing and logging utilities
 *
 * Provides:
 * - Configurable log levels (trace, debug, info, warn, error)
 * - Structured logging with timestamps and file/line info
 * - Performance timing measurements
 * - Environment variable control
 * - Thread-safe output
 *
 * Control via environment variable: ANOFOX_LOG_LEVEL
 * Values: trace, debug, info (default), warn, error
 *
 * Example usage:
 *   ANOFOX_TRACE(debug, "Processing " << rows << " rows");
 *   ANOFOX_TIMING_START();
 *   // ... do work ...
 *   ANOFOX_TIMING_END("Some operation");
 */

enum class LogLevel { TRACE = 0, DBG = 1, INFO = 2, WARN = 3, ERR = 4, NONE = 5 };

class Tracer {
public:
	/**
	 * @brief Initialize tracing system
	 *
	 * Reads ANOFOX_LOG_LEVEL environment variable
	 */
	static void Initialize();

	/**
	 * @brief Set global log level
	 *
	 * @param level Minimum level to output
	 */
	static void SetLogLevel(LogLevel level);

	/**
	 * @brief Get current log level
	 *
	 * @return Current LogLevel
	 */
	static LogLevel GetLogLevel();

	/**
	 * @brief Check if a message at given level should be logged
	 *
	 * @param level Message level
	 * @return true if should output
	 */
	static bool ShouldLog(LogLevel level);

	/**
	 * @brief Log a message with location information
	 *
	 * @param level Message level
	 * @param file Source file name
	 * @param line Source line number
	 * @param message Message content
	 */
	static void Log(LogLevel level, const std::string &file, int line, const std::string &message);

	/**
	 * @brief Log a message with level and location
	 *
	 * @param level Message level
	 * @param message Message content
	 */
	static void LogDirect(LogLevel level, const std::string &message);

	/**
	 * @brief Get string representation of log level
	 *
	 * @param level LogLevel to stringify
	 * @return String representation
	 */
	static std::string GetLevelName(LogLevel level);

	/**
	 * @brief Get timestamp string
	 *
	 * @return Current time as formatted string
	 */
	static std::string GetTimestamp();

	/**
	 * @brief Start a timed operation
	 *
	 * @return Opaque handle for timing
	 */
	static uint64_t TimingStart();

	/**
	 * @brief End a timed operation and log duration
	 *
	 * @param handle Handle from TimingStart()
	 * @param operation_name Human-readable operation name
	 * @return Duration in milliseconds
	 */
	static double TimingEnd(uint64_t handle, const std::string &operation_name);

private:
	static LogLevel current_level_;
	static bool initialized_;

	Tracer() = delete;
	~Tracer() = delete;
};

// ============================================================================
// Convenience Macros for Logging
// ============================================================================

/**
 * @brief Macro for trace-level logging with stream syntax
 *
 * Usage: ANOFOX_TRACE(message << stream << contents)
 */
#define ANOFOX_TRACE(msg)                                                                                              \
	do {                                                                                                               \
		if (duckdb::anofox_statistics::Tracer::ShouldLog(duckdb::anofox_statistics::LogLevel::TRACE)) {                \
			std::ostringstream oss;                                                                                    \
			oss << msg;                                                                                                \
			duckdb::anofox_statistics::Tracer::Log(duckdb::anofox_statistics::LogLevel::TRACE, __FILE__, __LINE__,     \
			                                       oss.str());                                                         \
		}                                                                                                              \
	} while (0)

/**
 * @brief Macro for debug-level logging with stream syntax
 *
 * Usage: ANOFOX_DEBUG(message << stream << contents)
 */
#define ANOFOX_DEBUG(msg)                                                                                              \
	do {                                                                                                               \
		if (duckdb::anofox_statistics::Tracer::ShouldLog(duckdb::anofox_statistics::LogLevel::DBG)) {                \
			std::ostringstream oss;                                                                                    \
			oss << msg;                                                                                                \
			duckdb::anofox_statistics::Tracer::Log(duckdb::anofox_statistics::LogLevel::DBG, __FILE__, __LINE__,     \
			                                       oss.str());                                                         \
		}                                                                                                              \
	} while (0)

/**
 * @brief Macro for info-level logging with stream syntax
 *
 * Usage: ANOFOX_INFO(message << stream << contents)
 */
#define ANOFOX_INFO(msg)                                                                                               \
	do {                                                                                                               \
		if (duckdb::anofox_statistics::Tracer::ShouldLog(duckdb::anofox_statistics::LogLevel::INFO)) {                 \
			std::ostringstream oss;                                                                                    \
			oss << msg;                                                                                                \
			duckdb::anofox_statistics::Tracer::Log(duckdb::anofox_statistics::LogLevel::INFO, __FILE__, __LINE__,      \
			                                       oss.str());                                                         \
		}                                                                                                              \
	} while (0)

/**
 * @brief Macro for warning-level logging with stream syntax
 *
 * Usage: ANOFOX_WARN(message << stream << contents)
 */
#define ANOFOX_WARN(msg)                                                                                               \
	do {                                                                                                               \
		if (duckdb::anofox_statistics::Tracer::ShouldLog(duckdb::anofox_statistics::LogLevel::WARN)) {                 \
			std::ostringstream oss;                                                                                    \
			oss << msg;                                                                                                \
			duckdb::anofox_statistics::Tracer::Log(duckdb::anofox_statistics::LogLevel::WARN, __FILE__, __LINE__,      \
			                                       oss.str());                                                         \
		}                                                                                                              \
	} while (0)

/**
 * @brief Macro for error-level logging with stream syntax
 *
 * Usage: ANOFOX_ERROR(message << stream << contents)
 */
#define ANOFOX_ERROR(msg)                                                                                              \
	do {                                                                                                               \
		std::ostringstream oss;                                                                                        \
		oss << msg;                                                                                                    \
		duckdb::anofox_statistics::Tracer::Log(duckdb::anofox_statistics::LogLevel::ERR, __FILE__, __LINE__,         \
		                                       oss.str());                                                             \
	} while (0)

/**
 * @brief Macro for timing operations
 *
 * Usage:
 *   ANOFOX_TIMING_START();
 *   // ... do work ...
 *   ANOFOX_TIMING_END("Operation name");
 */
#define ANOFOX_TIMING_START() uint64_t __anofox_timing_handle = duckdb::anofox_statistics::Tracer::TimingStart()

#define ANOFOX_TIMING_END(operation_name)                                                                              \
	duckdb::anofox_statistics::Tracer::TimingEnd(__anofox_timing_handle, operation_name)

} // namespace anofox_statistics
} // namespace duckdb
