#include "memory_manager.hpp"
#include "duckdb/common/exception.hpp"
#include <iostream>
#include <sstream>

namespace duckdb {
namespace anofox_statistics {

// ============================================================================
// MatrixBuffer Implementation
// ============================================================================

MemoryManager::MatrixBuffer::MatrixBuffer(size_t rows, size_t cols)
    : matrix_(new Eigen::MatrixXd(rows, cols)), allocated_bytes_(rows * cols * sizeof(double)) {
	MemoryManager::Get().RecordAllocation(allocated_bytes_);
}

MemoryManager::MatrixBuffer::MatrixBuffer(MatrixBuffer &&other) noexcept
    : matrix_(std::move(other.matrix_)), allocated_bytes_(other.allocated_bytes_) {
	other.allocated_bytes_ = 0;
}

MemoryManager::MatrixBuffer &MemoryManager::MatrixBuffer::operator=(MatrixBuffer &&other) noexcept {
	if (this != &other) {
		// Clean up current matrix if exists
		if (matrix_) {
			MemoryManager::Get().RecordDeallocation(allocated_bytes_);
		}
		matrix_ = std::move(other.matrix_);
		allocated_bytes_ = other.allocated_bytes_;
		other.allocated_bytes_ = 0;
	}
	return *this;
}

MemoryManager::MatrixBuffer::~MatrixBuffer() {
	if (matrix_) {
		MemoryManager::Get().RecordDeallocation(allocated_bytes_);
	}
}

Eigen::MatrixXd &MemoryManager::MatrixBuffer::Get() {
	if (!matrix_) {
		throw InvalidInputException("MatrixBuffer: Invalid buffer");
	}
	return *matrix_;
}

const Eigen::MatrixXd &MemoryManager::MatrixBuffer::Get() const {
	if (!matrix_) {
		throw InvalidInputException("MatrixBuffer: Invalid buffer");
	}
	return *matrix_;
}

Eigen::MatrixXd *MemoryManager::MatrixBuffer::operator->() {
	if (!matrix_) {
		throw InvalidInputException("MatrixBuffer: Invalid buffer");
	}
	return matrix_.get();
}

const Eigen::MatrixXd *MemoryManager::MatrixBuffer::operator->() const {
	if (!matrix_) {
		throw InvalidInputException("MatrixBuffer: Invalid buffer");
	}
	return matrix_.get();
}

Eigen::MatrixXd &MemoryManager::MatrixBuffer::operator*() {
	return Get();
}

const Eigen::MatrixXd &MemoryManager::MatrixBuffer::operator*() const {
	return Get();
}

bool MemoryManager::MatrixBuffer::IsValid() const {
	return matrix_ != nullptr;
}

size_t MemoryManager::MatrixBuffer::GetAllocatedBytes() const {
	return allocated_bytes_;
}

void MemoryManager::MatrixBuffer::Resize(size_t rows, size_t cols) {
	if (matrix_) {
		MemoryManager::Get().RecordDeallocation(allocated_bytes_);
	}
	matrix_.reset(new Eigen::MatrixXd(rows, cols));
	allocated_bytes_ = rows * cols * sizeof(double);
	MemoryManager::Get().RecordAllocation(allocated_bytes_);
}

// ============================================================================
// VectorBuffer Implementation
// ============================================================================

MemoryManager::VectorBuffer::VectorBuffer(size_t size)
    : vector_(new Eigen::VectorXd(size)), allocated_bytes_(size * sizeof(double)) {
	MemoryManager::Get().RecordAllocation(allocated_bytes_);
}

MemoryManager::VectorBuffer::VectorBuffer(VectorBuffer &&other) noexcept
    : vector_(std::move(other.vector_)), allocated_bytes_(other.allocated_bytes_) {
	other.allocated_bytes_ = 0;
}

MemoryManager::VectorBuffer &MemoryManager::VectorBuffer::operator=(VectorBuffer &&other) noexcept {
	if (this != &other) {
		if (vector_) {
			MemoryManager::Get().RecordDeallocation(allocated_bytes_);
		}
		vector_ = std::move(other.vector_);
		allocated_bytes_ = other.allocated_bytes_;
		other.allocated_bytes_ = 0;
	}
	return *this;
}

MemoryManager::VectorBuffer::~VectorBuffer() {
	if (vector_) {
		MemoryManager::Get().RecordDeallocation(allocated_bytes_);
	}
}

Eigen::VectorXd &MemoryManager::VectorBuffer::Get() {
	if (!vector_) {
		throw InvalidInputException("VectorBuffer: Invalid buffer");
	}
	return *vector_;
}

const Eigen::VectorXd &MemoryManager::VectorBuffer::Get() const {
	if (!vector_) {
		throw InvalidInputException("VectorBuffer: Invalid buffer");
	}
	return *vector_;
}

Eigen::VectorXd *MemoryManager::VectorBuffer::operator->() {
	if (!vector_) {
		throw InvalidInputException("VectorBuffer: Invalid buffer");
	}
	return vector_.get();
}

const Eigen::VectorXd *MemoryManager::VectorBuffer::operator->() const {
	if (!vector_) {
		throw InvalidInputException("VectorBuffer: Invalid buffer");
	}
	return vector_.get();
}

Eigen::VectorXd &MemoryManager::VectorBuffer::operator*() {
	return Get();
}

const Eigen::VectorXd &MemoryManager::VectorBuffer::operator*() const {
	return Get();
}

bool MemoryManager::VectorBuffer::IsValid() const {
	return vector_ != nullptr;
}

size_t MemoryManager::VectorBuffer::GetAllocatedBytes() const {
	return allocated_bytes_;
}

void MemoryManager::VectorBuffer::Resize(size_t size) {
	if (vector_) {
		MemoryManager::Get().RecordDeallocation(allocated_bytes_);
	}
	vector_.reset(new Eigen::VectorXd(size));
	allocated_bytes_ = size * sizeof(double);
	MemoryManager::Get().RecordAllocation(allocated_bytes_);
}

// ============================================================================
// MemoryManager Implementation
// ============================================================================

MemoryManager &MemoryManager::Get() {
	static MemoryManager instance;
	return instance;
}

MemoryManager::MatrixBuffer MemoryManager::CreateMatrixBuffer(size_t rows, size_t cols) {
	MemoryManager &manager = Get();

	size_t requested_bytes = rows * cols * sizeof(double);
	if (manager.WouldExceedLimit(requested_bytes)) {
		std::ostringstream oss;
		oss << "Memory allocation would exceed limit: "
		    << "requested " << requested_bytes << " bytes, "
		    << "limit " << manager.GetMemoryLimit() << " bytes, "
		    << "current " << manager.GetTotalAllocatedMemory() << " bytes";
		throw InvalidInputException(oss.str());
	}

	return MatrixBuffer(rows, cols);
}

MemoryManager::VectorBuffer MemoryManager::CreateVectorBuffer(size_t size) {
	MemoryManager &manager = Get();

	size_t requested_bytes = size * sizeof(double);
	if (manager.WouldExceedLimit(requested_bytes)) {
		std::ostringstream oss;
		oss << "Memory allocation would exceed limit: "
		    << "requested " << requested_bytes << " bytes, "
		    << "limit " << manager.GetMemoryLimit() << " bytes, "
		    << "current " << manager.GetTotalAllocatedMemory() << " bytes";
		throw InvalidInputException(oss.str());
	}

	return VectorBuffer(size);
}

size_t MemoryManager::GetTotalAllocatedMemory() const {
	return total_allocated_;
}

void MemoryManager::ResetStatistics() {
	total_allocated_ = 0;
	peak_allocated_ = 0;
}

size_t MemoryManager::GetPeakAllocatedMemory() const {
	return peak_allocated_;
}

void MemoryManager::SetMemoryLimit(size_t max_bytes) {
	memory_limit_ = max_bytes;
}

size_t MemoryManager::GetMemoryLimit() const {
	return memory_limit_;
}

bool MemoryManager::WouldExceedLimit(size_t requested_bytes) const {
	return (total_allocated_ + requested_bytes) > memory_limit_;
}

void MemoryManager::RecordAllocation(size_t bytes) {
	total_allocated_ += bytes;
	if (total_allocated_ > peak_allocated_) {
		peak_allocated_ = total_allocated_;
	}
}

void MemoryManager::RecordDeallocation(size_t bytes) {
	if (total_allocated_ >= bytes) {
		total_allocated_ -= bytes;
	}
}

} // namespace anofox_statistics
} // namespace duckdb
