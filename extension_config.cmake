# Extension configuration for anofox-statistics
# This extension provides statistical regression functions backed by the AnofoxStatistics C++ library

# Fix for older glibc versions (< 2.17) - add missing madvise constants
# This must be set BEFORE any DuckDB components are built
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -DMADV_DONTDUMP=24 -DMADV_DODUMP=25" CACHE STRING "C flags" FORCE)

duckdb_extension_load(anofox_statistics
    SOURCE_DIR ${CMAKE_CURRENT_LIST_DIR}
    LOAD_TESTS
)
