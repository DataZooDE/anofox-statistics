#pragma once
#include "duckdb.hpp"
#include "anofox_stats_ffi.h"
#include <string>

namespace duckdb {

/**
 * FFI error policy — the single place that decides what an FFI failure means
 * at the SQL level. Every aggregate, scalar, table and window function routes
 * its FFI errors through here.
 *
 *   Error code                        Kind            SQL outcome
 *   --------------------------------  --------------  ---------------------------
 *   INVALID_INPUT                     input error     InvalidInputException
 *   INVALID_ALPHA                     input error     InvalidInputException
 *   INVALID_L1_RATIO                  input error     InvalidInputException
 *   DIMENSION_MISMATCH                input error     InvalidInputException
 *   INSUFFICIENT_DATA                 degenerate      NULL
 *   NO_VALID_DATA                     degenerate      NULL
 *   SINGULAR_MATRIX                   degenerate      NULL
 *   CONVERGENCE_FAILURE               degenerate      NULL
 *   ALLOCATION_FAILURE                internal        OutOfMemoryException
 *   SERIALIZATION_ERROR, INTERNAL,    internal        InvalidInputException,
 *   unknown codes                                     message prefixed "internal error"
 *
 * "Input error" means the user's arguments or options are invalid (bad option
 * value, mismatched list lengths, impossible counts, ...): that is raised so it
 * is not mistaken for missing data. "Degenerate" means the data, although valid,
 * does not support a result (too few rows, everything NULL, a singular design,
 * a numerically broken fit): that is a NULL, like SQL's own aggregates on empty
 * input. GLM non-convergence is not an error at all — the fit is returned with
 * `converged = false`.
 *
 * Internal failures (a caught Rust panic, an unclassified upstream error) are
 * raised rather than hidden, but deliberately not as InternalException: DuckDB
 * treats that as an assertion failure of its own and may invalidate the
 * database.
 *
 * Message format: "<fn_name>: <error.message>". The two-arg printf form
 * ("%s", msg) is used so that a literal '%' in a Rust error message is never
 * treated as a format specifier.
 */
enum class FfiErrorKind { INPUT_ERROR, DEGENERATE, INTERNAL };

static inline FfiErrorKind ClassifyFfiError(AnofoxErrorCode code) {
    switch (code) {
        case ANOFOX_ERROR_INVALID_INPUT:
        case ANOFOX_ERROR_INVALID_ALPHA:
        case ANOFOX_ERROR_INVALID_L1_RATIO:
        case ANOFOX_ERROR_DIMENSION_MISMATCH:
            return FfiErrorKind::INPUT_ERROR;
        case ANOFOX_ERROR_INSUFFICIENT_DATA:
        case ANOFOX_ERROR_NO_VALID_DATA:
        case ANOFOX_ERROR_SINGULAR_MATRIX:
        case ANOFOX_ERROR_CONVERGENCE_FAILURE:
            return FfiErrorKind::DEGENERATE;
        default:
            return FfiErrorKind::INTERNAL;
    }
}

/**
 * Raise the exception for a non-degenerate FFI error. Used directly only where
 * a NULL result is impossible; everything else goes through
 * ThrowUnlessDegenerate.
 */
[[noreturn]] static inline void ThrowFromFfiError(const char *fn_name, const AnofoxError &err) {
    std::string msg = std::string(fn_name) + ": " + std::string(err.message);
    if (err.code == ANOFOX_ERROR_ALLOCATION_FAILURE) {
        throw OutOfMemoryException("%s", msg.c_str());
    }
    if (ClassifyFfiError(err.code) == FfiErrorKind::INTERNAL) {
        msg = std::string(fn_name) + ": internal error: " + std::string(err.message);
    }
    throw InvalidInputException("%s", msg.c_str());
}

/**
 * Apply the policy after a failed FFI call: throws for input and internal
 * errors, returns normally for degenerate outcomes so the caller can emit NULL.
 *
 *     if (!success) {
 *         ThrowUnlessDegenerate("ridge_fit_agg", error);
 *         FlatVector::SetNull(result, idx, true);
 *         continue;
 *     }
 */
static inline void ThrowUnlessDegenerate(const char *fn_name, const AnofoxError &err) {
    if (ClassifyFfiError(err.code) != FfiErrorKind::DEGENERATE) {
        ThrowFromFfiError(fn_name, err);
    }
}

} // namespace duckdb
