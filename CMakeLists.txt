cmake_minimum_required(VERSION 3.20)

set(TARGET_NAME anofox_stats)
set(EXTENSION_NAME ${TARGET_NAME}_extension)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

project(${TARGET_NAME})

# Corrosion for Rust integration
include(FetchContent)
FetchContent_Declare(
    Corrosion
    GIT_REPOSITORY https://github.com/corrosion-rs/corrosion.git
    GIT_TAG v0.5.2
)
set(CORROSION_VERBOSE_OUTPUT ON)
FetchContent_MakeAvailable(Corrosion)

# Detect Rust target based on platform
if(APPLE)
    if(CMAKE_OSX_ARCHITECTURES STREQUAL "arm64" OR CMAKE_SYSTEM_PROCESSOR STREQUAL "arm64")
        set(RUST_TARGET "aarch64-apple-darwin")
    else()
        set(RUST_TARGET "x86_64-apple-darwin")
    endif()
elseif(UNIX AND NOT APPLE)
    if(CMAKE_SYSTEM_PROCESSOR STREQUAL "aarch64")
        set(RUST_TARGET "aarch64-unknown-linux-gnu")
    else()
        set(RUST_TARGET "x86_64-unknown-linux-gnu")
    endif()
elseif(WIN32)
    if(CMAKE_SYSTEM_PROCESSOR STREQUAL "AMD64" OR CMAKE_SYSTEM_PROCESSOR STREQUAL "x86_64")
        set(RUST_TARGET "x86_64-pc-windows-msvc")
    else()
        set(RUST_TARGET "aarch64-pc-windows-msvc")
    endif()
endif()

message(STATUS "Detected Rust target: ${RUST_TARGET}")

# Import Rust crate - use full workspace path
corrosion_import_crate(
    MANIFEST_PATH ${CMAKE_CURRENT_SOURCE_DIR}/Cargo.toml
    CRATES anofox-stats-ffi
)

# Get the Rust library path
# Corrosion mangles crate names (hyphens to underscores)
set(RUST_LIB_PATH ${CMAKE_CURRENT_BINARY_DIR}/libanofox_stats_ffi.a)

# Extension source files
set(EXTENSION_SOURCES
    src/anofox_stats_extension.cpp
    src/table_functions/ols_fit.cpp
    src/table_functions/ridge_fit.cpp
    src/table_functions/elasticnet_fit.cpp
    src/aggregate_functions/ols_aggregate.cpp
    src/aggregate_functions/ridge_aggregate.cpp
    src/aggregate_functions/elasticnet_aggregate.cpp
)

# Check if we're being built as an out-of-tree extension
if(NOT "${DUCKDB_EXTENSION_MAIN_BUILD}")
    # Standalone build - need to find/build DuckDB
    message(STATUS "Building as standalone extension")

    # Include DuckDB
    if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/duckdb/CMakeLists.txt)
        set(EXTENSION_STATIC_BUILD TRUE)
        set(BUILD_SHELL FALSE)
        set(BUILD_UNITTESTS FALSE)
        set(BUILD_PARQUET_EXTENSION FALSE CACHE BOOL "" FORCE)
        set(BUILD_HTTPFS_EXTENSION FALSE CACHE BOOL "" FORCE)
        set(BUILD_JSON_EXTENSION FALSE CACHE BOOL "" FORCE)
        set(BUILD_TPCH_EXTENSION FALSE CACHE BOOL "" FORCE)
        set(BUILD_TPCDS_EXTENSION FALSE CACHE BOOL "" FORCE)
        set(BUILD_AUTOCOMPLETE_EXTENSION FALSE CACHE BOOL "" FORCE)
        set(BUILD_FTS_EXTENSION FALSE CACHE BOOL "" FORCE)
        add_subdirectory(duckdb)
    else()
        message(FATAL_ERROR "DuckDB source not found. Please clone DuckDB into ${CMAKE_CURRENT_SOURCE_DIR}/duckdb")
    endif()

    include_directories(duckdb/src/include)
endif()

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/src/include)

# Build the DuckDB loadable extension
build_loadable_extension(${TARGET_NAME} " " ${EXTENSION_SOURCES})
target_link_libraries(${TARGET_NAME}_loadable_extension ${RUST_LIB_PATH})
add_dependencies(${TARGET_NAME}_loadable_extension _cargo-build_anofox_stats_ffi)

# Build the static library extension
build_static_extension(${TARGET_NAME} ${EXTENSION_SOURCES})
target_link_libraries(${TARGET_NAME}_extension ${RUST_LIB_PATH})
add_dependencies(${TARGET_NAME}_extension _cargo-build_anofox_stats_ffi)
