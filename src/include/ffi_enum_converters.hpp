#pragma once

#include "anofox_stats_ffi.h"
#include "map_options_parser.hpp"

namespace duckdb {

/**
 * Convert C++ SolverType enum to FFI AnofoxSolverType.
 */
inline AnofoxSolverType ConvertSolverType(SolverType solver) {
    switch (solver) {
    case SolverType::QR:
        return ANOFOX_SOLVER_QR;
    case SolverType::SVD:
        return ANOFOX_SOLVER_SVD;
    case SolverType::CHOLESKY:
        return ANOFOX_SOLVER_CHOLESKY;
    default:
        return ANOFOX_SOLVER_SVD;
    }
}

/**
 * Convert C++ HcType enum to FFI AnofoxHcType.
 */
inline AnofoxHcType ConvertHcType(HcType hc) {
    switch (hc) {
    case HcType::NONE:
        return ANOFOX_HC_NONE;
    case HcType::HC0:
        return ANOFOX_HC_HC0;
    case HcType::HC1:
        return ANOFOX_HC_HC1;
    case HcType::HC2:
        return ANOFOX_HC_HC2;
    case HcType::HC3:
        return ANOFOX_HC_HC3;
    default:
        return ANOFOX_HC_NONE;
    }
}

/**
 * Convert C++ LambdaScaling enum to FFI AnofoxLambdaScaling.
 */
inline AnofoxLambdaScaling ConvertLambdaScaling(LambdaScaling scaling) {
    switch (scaling) {
    case LambdaScaling::RAW:
        return ANOFOX_LAMBDA_SCALING_RAW;
    case LambdaScaling::GLMNET:
        return ANOFOX_LAMBDA_SCALING_GLMNET;
    default:
        return ANOFOX_LAMBDA_SCALING_RAW;
    }
}

} // namespace duckdb
