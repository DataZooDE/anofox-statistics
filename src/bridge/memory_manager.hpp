#pragma once

#include <Eigen/Dense>
#include <vector>
#include <memory>
#include <cstddef>

namespace duckdb {
namespace anofox_statistics {

/**
 * @brief Memory management utilities for bridge layer
 *
 * Provides:
 * - Efficient allocation and pooling strategies
 * - RAII-based cleanup for temporary matrices
 * - Buffer reuse across multiple operations
 * - Zero-copy optimizations where possible
 */
class MemoryManager {
public:
	/**
	 * @brief RAII wrapper for temporary Eigen matrices
	 *
	 * Automatically deallocates memory when scope exits.
	 * Designed for use in table function bind phase.
	 */
	class MatrixBuffer {
	public:
		/**
		 * @brief Create a matrix buffer of specified dimensions
		 *
		 * @param rows Number of rows
		 * @param cols Number of columns
		 */
		MatrixBuffer(size_t rows, size_t cols);

		/**
		 * @brief Move constructor for efficient transfer
		 */
		MatrixBuffer(MatrixBuffer &&other) noexcept;

		/**
		 * @brief Move assignment
		 */
		MatrixBuffer &operator=(MatrixBuffer &&other) noexcept;

		/**
		 * @brief Destructor - automatically cleans up
		 */
		~MatrixBuffer();

		/**
		 * @brief Get mutable reference to underlying matrix
		 */
		Eigen::MatrixXd &Get();

		/**
		 * @brief Get const reference to underlying matrix
		 */
		const Eigen::MatrixXd &Get() const;

		/**
		 * @brief Get mutable pointer for direct access
		 */
		Eigen::MatrixXd *operator->();

		/**
		 * @brief Get const pointer for direct access
		 */
		const Eigen::MatrixXd *operator->() const;

		/**
		 * @brief Get mutable reference via dereference
		 */
		Eigen::MatrixXd &operator*();

		/**
		 * @brief Get const reference via dereference
		 */
		const Eigen::MatrixXd &operator*() const;

		/**
		 * @brief Check if buffer is valid
		 */
		bool IsValid() const;

		/**
		 * @brief Get number of allocated bytes
		 */
		size_t GetAllocatedBytes() const;

		/**
		 * @brief Reset to new dimensions
		 *
		 * @param rows New number of rows
		 * @param cols New number of columns
		 */
		void Resize(size_t rows, size_t cols);

	private:
		std::unique_ptr<Eigen::MatrixXd> matrix_;
		size_t allocated_bytes_;

		// Disable copy
		MatrixBuffer(const MatrixBuffer &) = delete;
		MatrixBuffer &operator=(const MatrixBuffer &) = delete;
	};

	/**
	 * @brief RAII wrapper for temporary Eigen vectors
	 */
	class VectorBuffer {
	public:
		/**
		 * @brief Create a vector buffer of specified size
		 *
		 * @param size Number of elements
		 */
		explicit VectorBuffer(size_t size);

		/**
		 * @brief Move constructor
		 */
		VectorBuffer(VectorBuffer &&other) noexcept;

		/**
		 * @brief Move assignment
		 */
		VectorBuffer &operator=(VectorBuffer &&other) noexcept;

		/**
		 * @brief Destructor
		 */
		~VectorBuffer();

		/**
		 * @brief Get mutable reference to underlying vector
		 */
		Eigen::VectorXd &Get();

		/**
		 * @brief Get const reference to underlying vector
		 */
		const Eigen::VectorXd &Get() const;

		/**
		 * @brief Get mutable pointer for direct access
		 */
		Eigen::VectorXd *operator->();

		/**
		 * @brief Get const pointer for direct access
		 */
		const Eigen::VectorXd *operator->() const;

		/**
		 * @brief Get mutable reference via dereference
		 */
		Eigen::VectorXd &operator*();

		/**
		 * @brief Get const reference via dereference
		 */
		const Eigen::VectorXd &operator*() const;

		/**
		 * @brief Check if buffer is valid
		 */
		bool IsValid() const;

		/**
		 * @brief Get number of allocated bytes
		 */
		size_t GetAllocatedBytes() const;

		/**
		 * @brief Reset to new size
		 *
		 * @param size New number of elements
		 */
		void Resize(size_t size);

	private:
		std::unique_ptr<Eigen::VectorXd> vector_;
		size_t allocated_bytes_;

		// Disable copy
		VectorBuffer(const VectorBuffer &) = delete;
		VectorBuffer &operator=(const VectorBuffer &) = delete;
	};

public:
	/**
	 * @brief Get singleton instance
	 *
	 * @return Reference to global MemoryManager
	 */
	static MemoryManager &Get();

	/**
	 * @brief Create a new matrix buffer
	 *
	 * @param rows Number of rows
	 * @param cols Number of columns
	 * @return MatrixBuffer RAII wrapper
	 */
	static MatrixBuffer CreateMatrixBuffer(size_t rows, size_t cols);

	/**
	 * @brief Create a new vector buffer
	 *
	 * @param size Number of elements
	 * @return VectorBuffer RAII wrapper
	 */
	static VectorBuffer CreateVectorBuffer(size_t size);

	/**
	 * @brief Get total allocated memory
	 *
	 * @return Size in bytes
	 */
	size_t GetTotalAllocatedMemory() const;

	/**
	 * @brief Reset memory statistics
	 */
	void ResetStatistics();

	/**
	 * @brief Get peak allocated memory
	 *
	 * @return Size in bytes
	 */
	size_t GetPeakAllocatedMemory() const;

	/**
	 * @brief Configure memory limits for safety
	 *
	 * @param max_bytes Maximum bytes to allocate before throwing
	 */
	void SetMemoryLimit(size_t max_bytes);

	/**
	 * @brief Get configured memory limit
	 *
	 * @return Maximum bytes allowed
	 */
	size_t GetMemoryLimit() const;

	/**
	 * @brief Check if allocation would exceed limit
	 *
	 * @param requested_bytes Bytes to be allocated
	 * @return true if would exceed limit
	 */
	bool WouldExceedLimit(size_t requested_bytes) const;

private:
	MemoryManager() = default;
	~MemoryManager() = default;

	size_t total_allocated_ = 0;
	size_t peak_allocated_ = 0;
	size_t memory_limit_ = 1024 * 1024 * 1024; // 1GB default

	void RecordAllocation(size_t bytes);
	void RecordDeallocation(size_t bytes);

	// Singleton
	MemoryManager(const MemoryManager &) = delete;
	MemoryManager &operator=(const MemoryManager &) = delete;

	friend class MatrixBuffer;
	friend class VectorBuffer;
};

} // namespace anofox_statistics
} // namespace duckdb
