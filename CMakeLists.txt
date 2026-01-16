cmake_minimum_required(VERSION 3.20)

set(CORROSION_VERBOSE_OUTPUT ON)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Get installed Rust targets
execute_process(
    COMMAND rustup target list --installed
    OUTPUT_VARIABLE RUST_TARGETS
    ERROR_QUIET
)

# Propagate arch to rust build for CI
set(Rust_CARGO_TARGET "")
if("${OS_NAME}" STREQUAL "linux")
    if ("${OS_ARCH}" STREQUAL "arm64")
        set(Rust_CARGO_TARGET "aarch64-unknown-linux-gnu")
    elseif("${CMAKE_CXX_COMPILER}" MATCHES "aarch64")
        set(Rust_CARGO_TARGET "aarch64-unknown-linux-gnu")
    else()
        string(FIND "${RUST_TARGETS}" "musl" MUSL_TARGET_FOUND)
        if(NOT MUSL_TARGET_FOUND EQUAL -1)
            set(Rust_CARGO_TARGET "x86_64-unknown-linux-musl")
        else()
            set(Rust_CARGO_TARGET "x86_64-unknown-linux-gnu")
        endif()
    endif()
elseif("${OS_NAME}" STREQUAL "osx")
    if ("${OSX_BUILD_ARCH}" STREQUAL "arm64")
        set(Rust_CARGO_TARGET "aarch64-apple-darwin")
    elseif ("${OSX_BUILD_ARCH}" STREQUAL "x86_64")
        set(Rust_CARGO_TARGET "x86_64-apple-darwin")
    elseif ("${OS_ARCH}" STREQUAL "arm64")
        set(Rust_CARGO_TARGET "aarch64-apple-darwin")
    endif()
elseif(WIN32)
    if (MINGW AND "${OS_ARCH}" STREQUAL "arm64")
        set(Rust_CARGO_TARGET "aarch64-pc-windows-gnu")
    elseif (MINGW AND "${OS_ARCH}" STREQUAL "amd64")
        set(Rust_CARGO_TARGET "x86_64-pc-windows-gnu")
    elseif (MSVC AND "${OS_ARCH}" STREQUAL "arm64")
        set(Rust_CARGO_TARGET "aarch64-pc-windows-msvc")
    elseif (MSVC AND "${OS_ARCH}" STREQUAL "amd64")
        set(Rust_CARGO_TARGET "x86_64-pc-windows-msvc")
    endif()
endif()

# Check for WASM target
string(FIND "${RUST_TARGETS}" "wasm32-unknown-emscripten" WASM_TARGET_FOUND)
if (NOT WASM_TARGET_FOUND EQUAL -1)
    set(Rust_CARGO_TARGET "wasm32-unknown-emscripten")
endif()

message(STATUS "Rust_CARGO_TARGET: ${Rust_CARGO_TARGET}")
message(STATUS "OS_NAME: ${OS_NAME}")
message(STATUS "OS_ARCH: ${OS_ARCH}")

# Fetch Corrosion for Rust integration
# Disable vcpkg for Corrosion's internal CMake to avoid toolchain conflicts
include(FetchContent)

FetchContent_Declare(
    Corrosion
    GIT_REPOSITORY https://github.com/corrosion-rs/corrosion.git
    GIT_TAG v0.5.2
)
# Set any global configuration variables such as `Rust_TOOLCHAIN` before this line!
FetchContent_MakeAvailable(Corrosion)

# Import the Rust FFI crate
corrosion_import_crate(
    MANIFEST_PATH "${CMAKE_CURRENT_SOURCE_DIR}/crates/anofox-stats-ffi/Cargo.toml"
    CRATES "anofox_stats_ffi"
)

# Set extension name here
set(TARGET_NAME anofox_statistics)
set(EXTENSION_NAME ${TARGET_NAME}_extension)
set(LOADABLE_EXTENSION_NAME ${TARGET_NAME}_loadable_extension)

project(${TARGET_NAME})

# --- Telemetry ---
find_package(OpenSSL REQUIRED)

add_library(posthog_telemetry_obj OBJECT
    ${CMAKE_CURRENT_SOURCE_DIR}/posthog-telemetry/src/telemetry.cpp
)

target_include_directories(posthog_telemetry_obj PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/posthog-telemetry/include
    ${CMAKE_CURRENT_SOURCE_DIR}/duckdb/src/include
    ${CMAKE_CURRENT_SOURCE_DIR}/duckdb/third_party
    ${CMAKE_CURRENT_SOURCE_DIR}/duckdb/third_party/httplib
    ${OPENSSL_INCLUDE_DIR}
)

target_compile_features(posthog_telemetry_obj PUBLIC cxx_std_17)

include_directories(src/include)

# Extension source files
set(EXTENSION_SOURCES
    src/anofox_statistics_extension.cpp
    src/include/map_options_parser.cpp
    src/table_functions/ols_fit.cpp
    src/table_functions/ridge_fit.cpp
    src/table_functions/elasticnet_fit.cpp
    src/table_functions/wls_fit.cpp
    src/table_functions/predict.cpp
    src/table_functions/rls_fit.cpp
    src/aggregate_functions/ols_aggregate.cpp
    src/aggregate_functions/ridge_aggregate.cpp
    src/aggregate_functions/elasticnet_aggregate.cpp
    src/aggregate_functions/wls_aggregate.cpp
    src/aggregate_functions/rls_aggregate.cpp
    src/aggregate_functions/vif_aggregate.cpp
    src/aggregate_functions/jarque_bera_aggregate.cpp
    src/aggregate_functions/residuals_diagnostics_aggregate.cpp
    src/aggregate_functions/ols_predict_aggregate.cpp
    src/aggregate_functions/ridge_predict_aggregate.cpp
    src/aggregate_functions/wls_predict_aggregate.cpp
    src/aggregate_functions/rls_predict_aggregate.cpp
    src/aggregate_functions/elasticnet_predict_aggregate.cpp
    src/aggregate_functions/poisson_aggregate.cpp
    src/aggregate_functions/alm_aggregate.cpp
    src/aggregate_functions/bls_aggregate.cpp
    src/aggregate_functions/bls_fit_predict_aggregate.cpp
    src/aggregate_functions/alm_fit_predict_aggregate.cpp
    src/aggregate_functions/poisson_fit_predict_aggregate.cpp
    src/aggregate_functions/aid_aggregate.cpp
    src/aggregate_functions/shapiro_wilk_aggregate.cpp
    src/aggregate_functions/t_test_aggregate.cpp
    src/aggregate_functions/pearson_aggregate.cpp
    src/aggregate_functions/spearman_aggregate.cpp
    src/aggregate_functions/mann_whitney_aggregate.cpp
    src/aggregate_functions/anova_aggregate.cpp
    src/aggregate_functions/kruskal_wallis_aggregate.cpp
    src/aggregate_functions/chisq_aggregate.cpp
    src/aggregate_functions/kendall_aggregate.cpp
    src/aggregate_functions/fisher_exact_aggregate.cpp
    src/aggregate_functions/brunner_munzel_aggregate.cpp
    src/aggregate_functions/dagostino_k2_aggregate.cpp
    src/aggregate_functions/energy_distance_aggregate.cpp
    src/aggregate_functions/mmd_aggregate.cpp
    src/aggregate_functions/tost_t_test_aggregate.cpp
    src/aggregate_functions/wilcoxon_signed_rank_aggregate.cpp
    src/aggregate_functions/distance_cor_aggregate.cpp
    src/aggregate_functions/yuen_aggregate.cpp
    src/aggregate_functions/brown_forsythe_aggregate.cpp
    src/aggregate_functions/diebold_mariano_aggregate.cpp
    src/aggregate_functions/clark_west_aggregate.cpp
    src/aggregate_functions/permutation_t_test_aggregate.cpp
    src/aggregate_functions/tost_paired_aggregate.cpp
    src/aggregate_functions/tost_correlation_aggregate.cpp
    src/aggregate_functions/chisq_gof_aggregate.cpp
    src/aggregate_functions/prop_test_one_aggregate.cpp
    src/aggregate_functions/prop_test_two_aggregate.cpp
    src/aggregate_functions/binom_test_aggregate.cpp
    src/aggregate_functions/cramers_v_aggregate.cpp
    src/aggregate_functions/cohen_kappa_aggregate.cpp
    src/aggregate_functions/icc_aggregate.cpp
    src/aggregate_functions/g_test_aggregate.cpp
    src/aggregate_functions/mcnemar_aggregate.cpp
    src/aggregate_functions/phi_coefficient_aggregate.cpp
    src/aggregate_functions/contingency_coef_aggregate.cpp
    src/aggregate_functions/pls_fit_predict_aggregate.cpp
    src/aggregate_functions/isotonic_fit_predict_aggregate.cpp
    src/aggregate_functions/quantile_fit_predict_aggregate.cpp
    src/scalar_functions/vif.cpp
    src/scalar_functions/aic_bic.cpp
    src/scalar_functions/jarque_bera.cpp
    src/scalar_functions/residuals_diagnostics.cpp
    src/window_functions/ols_fit_predict.cpp
    src/window_functions/ridge_fit_predict.cpp
    src/window_functions/wls_fit_predict.cpp
    src/window_functions/rls_fit_predict.cpp
    src/window_functions/elasticnet_fit_predict.cpp
    src/macros/fit_predict_macros.cpp
)

# Build static extension
build_static_extension(${TARGET_NAME} ${EXTENSION_SOURCES})

# Build loadable extension
build_loadable_extension(${TARGET_NAME} " " ${EXTENSION_SOURCES})

# Add telemetry objects to both extension targets
target_sources(${EXTENSION_NAME} PRIVATE $<TARGET_OBJECTS:posthog_telemetry_obj>)
target_sources(${LOADABLE_EXTENSION_NAME} PRIVATE $<TARGET_OBJECTS:posthog_telemetry_obj>)

# Add telemetry include path
target_include_directories(${EXTENSION_NAME} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/posthog-telemetry/include)
target_include_directories(${LOADABLE_EXTENSION_NAME} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/posthog-telemetry/include)

# Link to Rust library and OpenSSL (for telemetry)
target_link_libraries(${EXTENSION_NAME} anofox_stats_ffi-static OpenSSL::SSL OpenSSL::Crypto)
target_link_libraries(${LOADABLE_EXTENSION_NAME} anofox_stats_ffi-static OpenSSL::SSL OpenSSL::Crypto)

# Link Windows system libraries
if(WIN32)
    # bcrypt: required by Rust's getrandom crate
    # iphlpapi: required by telemetry GetAdaptersInfo
    target_link_libraries(${EXTENSION_NAME} bcrypt iphlpapi)
    target_link_libraries(${LOADABLE_EXTENSION_NAME} bcrypt iphlpapi)
endif()

# Install
install(
    TARGETS ${EXTENSION_NAME}
    EXPORT "${DUCKDB_EXPORT_SET}"
    LIBRARY DESTINATION "${INSTALL_LIB_DIR}"
    ARCHIVE DESTINATION "${INSTALL_LIB_DIR}"
)
