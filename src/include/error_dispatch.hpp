#pragma once
#include "duckdb.hpp"
#include "anofox_stats_ffi.h"
#include <string>

namespace duckdb {

/**
 * Dispatch an FFI error to the correct DuckDB exception type based on the
 * error code returned by the Rust layer.
 *
 * Taxonomy (locked in CONTEXT.md):
 *   Numerical/internal failures (SingularMatrix, ConvergenceFailure,
 *   AllocationFailure, Internal) → InternalException.  These are computation
 *   failures, not user data problems.  The CONTEXT.md intent is "FunctionException"
 *   but that class does not exist in the embedded DuckDB build; InternalException
 *   is the closest available type (ExceptionType::INTERNAL) and already used in
 *   this codebase for non-user-caused failures.
 *
 *   User data / shape problems (DimensionMismatch, InsufficientData, NoValidData,
 *   InvalidInput, InvalidAlpha, InvalidL1Ratio, SerializationError, any unknown
 *   code) → InvalidInputException.
 *
 * Message format: "<fn_name>: <error.message>" — always names the function.
 * The two-arg printf form ("%s", msg) is used so that a literal '%' in a
 * Rust error message is never treated as a format specifier.
 */
static inline void ThrowFromFfiError(const char *fn_name, const AnofoxError &err) {
    std::string msg = std::string(fn_name) + ": " + std::string(err.message);
    switch (err.code) {
        case ANOFOX_ERROR_SINGULAR_MATRIX:
        case ANOFOX_ERROR_CONVERGENCE_FAILURE:
        case ANOFOX_ERROR_INTERNAL:
        case ANOFOX_ERROR_ALLOCATION_FAILURE:
            // Numerical / internal failure — computation failed, not user data
            throw InternalException(msg);
        default:
            // InsufficientData, DimensionMismatch, InvalidInput, NoValidData,
            // InvalidAlpha, InvalidL1Ratio, SerializationError, and any
            // unrecognised code → user data / shape problem.
            throw InvalidInputException("%s", msg.c_str());
    }
}

} // namespace duckdb
