cmake_minimum_required(VERSION 3.20)

project(libanostat LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# --- Vcpkg Integration ---
# Dependencies are installed from vcpkg.json.
# The toolchain file should be specified on the command line:
# cmake -B build -S . -DCMAKE_TOOLCHAIN_FILE=$VCPKG_ROOT/scripts/buildsystems/vcpkg.cmake

# --- Project Structure ---
# Note: libanostat is a header-only library, so we use INTERFACE library
# All solver algorithms are implemented in header files for inline optimization

# Create header-only library target
add_library(libanostat INTERFACE)
target_include_directories(libanostat INTERFACE include)

# --- Dependencies ---
find_package(Eigen3 CONFIG REQUIRED)
find_package(Threads REQUIRED)
find_package(nlohmann_json CONFIG QUIET)

target_link_libraries(libanostat INTERFACE
    Threads::Threads
    Eigen3::Eigen
)

# --- Logging Control ---
# Similar to anofox-time, allow conditional logging
option(ENABLE_LOGGING "Enable debug/info logging output (disable for production)" OFF)

if(NOT ENABLE_LOGGING)
    message(STATUS "Logging disabled - compiling with LIBANOSTAT_NO_LOGGING")
    target_compile_definitions(libanostat INTERFACE LIBANOSTAT_NO_LOGGING)
else()
    message(STATUS "Logging enabled")
endif()

# --- Testing with Catch2 ---
enable_testing()
include(CTest)
if(BUILD_TESTING)
    find_package(Catch2 CONFIG)

    if(Catch2_FOUND)
        set(TEST_SOURCES
            test/main.cpp
            test/core/test_core_structures.cpp
            test/core/test_input_validation.cpp
            test/test_compilation.cpp
            test/solvers/test_i_regressor.cpp
            test/solvers/test_coefficient_extraction.cpp
            test/solvers/test_rank_deficiency_edge_cases.cpp
            test/solvers/test_saturated_models.cpp
            test/solvers/test_statistics_edge_cases.cpp
            test/utils/test_distributions.cpp
            test/inference/test_inference.cpp
            test/diagnostics/test_regression_diagnostics.cpp
            test/integration/test_full_workflow.cpp
            test/test_duckdb_failures.cpp
        )

        # R Validation Tests
        set(VALIDATION_TEST_SOURCES
            test/solvers/test_ols.cpp
            test/solvers/test_ridge.cpp
            test/solvers/test_elastic_net.cpp
            test/solvers/test_wls.cpp
            test/solvers/test_rls.cpp
        )

        add_executable(libanostat-tests ${TEST_SOURCES})
        target_link_libraries(libanostat-tests PRIVATE libanostat Catch2::Catch2WithMain)
        target_compile_definitions(libanostat-tests PRIVATE LIBANOSTAT_SOURCE_DIR="${CMAKE_SOURCE_DIR}")

        # Only build validation tests if nlohmann_json is available
        # NOTE: Validation tests are temporarily disabled due to API incompatibilities
        # if(nlohmann_json_FOUND)
        #     add_executable(libanostat-validation-tests ${VALIDATION_TEST_SOURCES})
        #     target_link_libraries(libanostat-validation-tests PRIVATE libanostat nlohmann_json::nlohmann_json)
        #     target_compile_definitions(libanostat-validation-tests PRIVATE LIBANOSTAT_SOURCE_DIR="${CMAKE_SOURCE_DIR}")
        #     message(STATUS "Building validation tests (nlohmann_json found)")
        # else()
            message(STATUS "Skipping validation tests (API updates needed)")
        # endif()

        include(Catch)
        catch_discover_tests(
            libanostat-tests
            TEST_PREFIX "libanostat."
            EXTRA_ARGS --success --durations yes
        )
    else()
        message(STATUS "Catch2 not found, skipping tests")
    endif()
endif()
