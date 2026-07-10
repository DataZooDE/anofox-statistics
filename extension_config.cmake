#This file is included by DuckDB's build system. It specifies which extension to load.
duckdb_extension_load(anofox_statistics
    SOURCE_DIR ${CMAKE_CURRENT_LIST_DIR}
    LOAD_TESTS
    # On WASM builds the extension target is a STATIC library and the final
    # link happens in a post-build emcc step (extension_build_tools.cmake:196)
    # that reads its extra archives from DUCKDB_EXTENSION_ANOFOX_STATISTICS_LINKED_LIBS.
    # Without this, the Rust FFI static archive (target: anofox_stats_ffi, defined
    # in CMakeLists.txt via corrosion_import_crate) is never linked into the .wasm,
    # so all C-linkage FFI symbols end up as unresolvable imports and LOAD fails
    # with "TypeError: r is not a function" in DuckDB-Wasm (issue #102, upstream
    # duckdb/duckdb#23740, sister-extension fix DataZooDE/anofox-forecast#240).
    # Corrosion registers two targets: `anofox_stats_ffi` (INTERFACE — for
    # `target_link_libraries`) and `anofox_stats_ffi-static` (the actual STATIC
    # IMPORTED archive). We need the -static suffix here so TARGET_FILE
    # resolves to the .a file path.
    LINKED_LIBS "$<TARGET_FILE:anofox_stats_ffi-static>"
)
