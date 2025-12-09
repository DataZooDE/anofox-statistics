cmake_minimum_required(VERSION 3.20)

    set(TARGET_NAME anofox_stats) set(EXTENSION_NAME ${
        TARGET_NAME} _extension) set(CMAKE_CXX_STANDARD 17) set(CMAKE_CXX_STANDARD_REQUIRED ON)

        project(${TARGET_NAME})

#The Rust library is pre - built by the Makefile(cargo build-- release)
#Just point to it directly
            set(RUST_LIB_PATH ${CMAKE_CURRENT_SOURCE_DIR} / target / release / libanofox_stats_ffi.a)

#Verify it exists
                if (NOT EXISTS ${RUST_LIB_PATH})
                    message(FATAL_ERROR
                            "Rust library not found at ${RUST_LIB_PATH}. Run 'cargo build --release' first.") endif()

                        message(STATUS "Using pre-built Rust library: ${RUST_LIB_PATH}")

#Extension source files
                            set(EXTENSION_SOURCES src / anofox_stats_extension.cpp src / table_functions /
                                ols_fit.cpp src / table_functions / ridge_fit.cpp src / table_functions /
                                elasticnet_fit.cpp src / table_functions / wls_fit.cpp src / table_functions /
                                predict.cpp src / table_functions / rls_fit.cpp src / aggregate_functions /
                                ols_aggregate.cpp src / aggregate_functions / ridge_aggregate.cpp src /
                                aggregate_functions / elasticnet_aggregate.cpp src / aggregate_functions /
                                wls_aggregate.cpp src / aggregate_functions / rls_aggregate.cpp src /
                                aggregate_functions / vif_aggregate.cpp src / aggregate_functions /
                                jarque_bera_aggregate.cpp src / aggregate_functions /
                                residuals_diagnostics_aggregate.cpp src / scalar_functions / vif.cpp src /
                                scalar_functions / aic_bic.cpp src / scalar_functions / jarque_bera.cpp src /
                                scalar_functions / residuals_diagnostics.cpp)

                                include_directories(${CMAKE_CURRENT_SOURCE_DIR} / src / include)

#Build the DuckDB loadable extension
                                    build_loadable_extension($ { TARGET_NAME } " " ${EXTENSION_SOURCES})
                                        target_link_libraries(${TARGET_NAME} _loadable_extension ${RUST_LIB_PATH})

#Build the static library extension(for linking into DuckDB)
                                            build_static_extension(${TARGET_NAME} ${EXTENSION_SOURCES})
                                                target_link_libraries(${TARGET_NAME} _extension ${RUST_LIB_PATH})

#Add to DuckDB export set(required for install / export to work)
                                                    install(TARGETS ${TARGET_NAME} _extension EXPORT "DuckDBExports")
