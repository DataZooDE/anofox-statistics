cmake_minimum_required(VERSION 3.20)

# Set extension name and version
set(TARGET_NAME anofox_statistics)
project(${TARGET_NAME})

# DuckDB extension boilerplate
include_directories(src/include)

# Find Eigen3 - required for compilation but optional for CMake configure (code quality checks)
find_package(Eigen3 QUIET CONFIG)
if(Eigen3_FOUND)
  if(DEFINED VCPKG_BUILD)
    message(STATUS "Found Eigen3 (vcpkg): ${Eigen3_DIR}")
  else()
    message(STATUS "Found Eigen3 (system): ${Eigen3_DIR}")
  endif()
else()
  message(WARNING "Eigen3 not found - extension will not compile without it. Install via vcpkg or system package (libeigen3-dev).")
endif()

# Extension sources - Phase 2 (Adding fit functions with Eigen/bridge layer)
set(EXTENSION_SOURCES
    # Core
    src/anofox_statistics_extension.cpp

    # Utils
    src/utils/tracing.cpp
    src/utils/validation.cpp  # Updated for DuckDB v1.4.1 API

    # Bridge layer (for Eigen integration)
    # src/bridge/type_converter.cpp  # TODO: Update for v1.4.1 if needed
    # src/bridge/memory_manager.cpp  # TODO: Update for v1.4.1 if needed

    # Functions - Metrics
    src/functions/ols_metrics.cpp

    # Functions - Fit (Phase 2 - Array-based implementation for v1.4.1)
    src/functions/ols_fit_v2.cpp   # Array-based OLS fit using DOUBLE[] inputs
    src/functions/ridge_fit.cpp    # Ridge regression with L2 regularization
    src/functions/wls_fit.cpp      # Weighted Least Squares

    # Functions - Online/Sequential (Phase 3)
    src/functions/rls_fit.cpp       # Recursive Least Squares (online learning)
    src/functions/rolling_ols.cpp   # Rolling window OLS (time-series)
    src/functions/expanding_ols.cpp # Expanding window OLS (cumulative)

    # Phase 4 - Aggregates (✅ completed)
    src/functions/aggregates/ols_aggregate.cpp  # OLS aggregate with GROUP BY

    # Phase 5 - Statistical Inference & Diagnostics (In Progress)
    src/functions/inference/ols_inference.cpp          # Coefficient inference with p-values
    src/functions/inference/prediction_intervals.cpp   # Confidence/prediction intervals
    src/functions/model_selection/information_criteria.cpp  # AIC, BIC, model selection
    src/functions/diagnostics/residual_diagnostics.cpp # Leverage, Cook's D, DFFITS
    src/functions/diagnostics/vif.cpp                  # Multicollinearity detection
    src/functions/diagnostics/normality_test.cpp       # Jarque-Bera normality test
)

# Build the static extension (required for DuckDB to link against when building statically)
build_static_extension(${TARGET_NAME} ${EXTENSION_SOURCES})
if(Eigen3_FOUND)
  target_link_libraries(${TARGET_NAME}_extension Eigen3::Eigen)
endif()

# Build the loadable extension
build_loadable_extension(${TARGET_NAME} "" ${EXTENSION_SOURCES})
if(Eigen3_FOUND)
  target_link_libraries(${TARGET_NAME}_loadable_extension Eigen3::Eigen)
endif()

# Install rules (installs the static extension target)
install(
    TARGETS ${TARGET_NAME}_extension
    EXPORT "${DUCKDB_EXPORT_SET}"
    LIBRARY DESTINATION "${INSTALL_LIB_DIR}"
    ARCHIVE DESTINATION "${INSTALL_LIB_DIR}")
