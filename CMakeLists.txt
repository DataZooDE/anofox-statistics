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
    CRATES "anofox-stats-ffi"
)

# Set extension name here
set(TARGET_NAME anofox_stats)
set(EXTENSION_NAME ${TARGET_NAME}_extension)
set(LOADABLE_EXTENSION_NAME ${TARGET_NAME}_loadable_extension)

project(${TARGET_NAME})

include_directories(src/include)

# Extension source files
set(EXTENSION_SOURCES
    src/anofox_stats_extension.cpp
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
    src/scalar_functions/vif.cpp
    src/scalar_functions/aic_bic.cpp
    src/scalar_functions/jarque_bera.cpp
    src/scalar_functions/residuals_diagnostics.cpp
)

# Build static extension
build_static_extension(${TARGET_NAME} ${EXTENSION_SOURCES})

# Build loadable extension
build_loadable_extension(${TARGET_NAME} " " ${EXTENSION_SOURCES})

# Link to Rust library (use static for both)
target_link_libraries(${EXTENSION_NAME} anofox-stats-ffi-static)
target_link_libraries(${LOADABLE_EXTENSION_NAME} anofox-stats-ffi-static)

# Install
install(
    TARGETS ${EXTENSION_NAME}
    EXPORT "${DUCKDB_EXPORT_SET}"
    LIBRARY DESTINATION "${INSTALL_LIB_DIR}"
    ARCHIVE DESTINATION "${INSTALL_LIB_DIR}"
)
