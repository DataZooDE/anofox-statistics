#include "tracing.hpp"
#include <cstdlib>
#include <mutex>

namespace duckdb {
namespace anofox_statistics {

// Static member initialization
// Release builds: default to WARN (suppress INFO messages)
// Debug builds: default to INFO (show all non-debug messages)
#ifdef NDEBUG
LogLevel Tracer::current_level_ = LogLevel::WARN;
#else
LogLevel Tracer::current_level_ = LogLevel::INFO;
#endif
bool Tracer::initialized_ = false;

// Global mutex for thread-safe logging
// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
static std::mutex g_tracer_mutex;

void Tracer::Initialize() {
	if (initialized_) {
		return;
	}

	initialized_ = true;

	const char *env_level = std::getenv("ANOFOX_LOG_LEVEL");
	if (env_level == nullptr) {
		// Use build-type-specific default if no env var is set
#ifdef NDEBUG
		current_level_ = LogLevel::WARN; // Release: suppress INFO messages
#else
		current_level_ = LogLevel::INFO; // Debug: show INFO messages
#endif
		return;
	}

	std::string level_str = env_level;

	// Convert to lowercase for comparison
	for (auto &c : level_str) {
		c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
	}

	if (level_str == "trace") {
		current_level_ = LogLevel::TRACE;
	} else if (level_str == "debug") {
		current_level_ = LogLevel::DBG;
	} else if (level_str == "info") {
		current_level_ = LogLevel::INFO;
	} else if (level_str == "warn") {
		current_level_ = LogLevel::WARN;
	} else if (level_str == "error") {
		current_level_ = LogLevel::ERR;
	} else if (level_str == "none") {
		current_level_ = LogLevel::NONE;
	} else {
		// Default if unrecognized - use build-type-specific default
#ifdef NDEBUG
		current_level_ = LogLevel::WARN;
#else
		current_level_ = LogLevel::INFO;
#endif
	}
}

void Tracer::SetLogLevel(LogLevel level) {
	current_level_ = level;
	initialized_ = true;
}

LogLevel Tracer::GetLogLevel() {
	if (!initialized_) {
		Initialize();
	}
	return current_level_;
}

bool Tracer::ShouldLog(LogLevel level) {
	if (!initialized_) {
		Initialize();
	}
	return level >= current_level_;
}

std::string Tracer::GetLevelName(LogLevel level) {
	switch (level) {
	case LogLevel::TRACE:
		return "TRACE";
	case LogLevel::DBG:
		return "DEBUG";
	case LogLevel::INFO:
		return "INFO";
	case LogLevel::WARN:
		return "WARN";
	case LogLevel::ERR:
		return "ERROR";
	case LogLevel::NONE:
		return "NONE";
	default:
		return "UNKNOWN";
	}
}

std::string Tracer::GetTimestamp() {
	auto now = std::chrono::system_clock::now();
	auto time = std::chrono::system_clock::to_time_t(now);
	auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(now.time_since_epoch()) % 1000;

	std::ostringstream oss;
	oss << std::put_time(std::localtime(&time), "%Y-%m-%d %H:%M:%S") << "." << std::setfill('0') << std::setw(3)
	    << ms.count();

	return oss.str();
}

void Tracer::Log(LogLevel level, const std::string &file, int line, const std::string &message) {
	if (!ShouldLog(level)) {
		return;
	}

	std::lock_guard<std::mutex> lock(g_tracer_mutex);

	std::string timestamp = GetTimestamp();
	std::string level_name = GetLevelName(level);

	// Extract filename from full path
	size_t last_slash = file.find_last_of("/\\");
	std::string filename = (last_slash == std::string::npos) ? file : file.substr(last_slash + 1);

	std::cerr << "[" << timestamp << "] [anofox/" << level_name << "] " << filename << ":" << line << " - " << message
	          << '\n';
}

void Tracer::LogDirect(LogLevel level, const std::string &message) {
	if (!ShouldLog(level)) {
		return;
	}

	std::lock_guard<std::mutex> lock(g_tracer_mutex);

	std::string timestamp = GetTimestamp();
	std::string level_name = GetLevelName(level);

	std::cerr << "[" << timestamp << "] [anofox/" << level_name << "] " << message << '\n';
}

uint64_t Tracer::TimingStart() {
	return static_cast<uint64_t>(std::chrono::high_resolution_clock::now().time_since_epoch().count());
}

double Tracer::TimingEnd(uint64_t handle, const std::string &operation_name) {
	auto end_time = std::chrono::high_resolution_clock::now().time_since_epoch();
	uint64_t end_ns = static_cast<uint64_t>(end_time.count());
	uint64_t duration_ns = end_ns - handle;
	double duration_ms = static_cast<double>(duration_ns) / 1000000.0;

	std::ostringstream oss;
	oss << std::fixed << std::setprecision(2);
	oss << operation_name << " completed in " << duration_ms << " ms";

	LogDirect(LogLevel::DBG, oss.str());

	return duration_ms;
}

} // namespace anofox_statistics
} // namespace duckdb
