.PHONY: all clean debug release test rust_release rust_debug rust_test

MKFILE_PATH := $(abspath $(lastword $(MAKEFILE_LIST)))
PROJ_DIR := $(dir $(MKFILE_PATH))

# Extension config
EXT_NAME=ANOFOX_STATS
EXT_CONFIG=${PROJ_DIR}extension_config.cmake

# Include the DuckDB extension makefile
include extension-ci-tools/makefiles/duckdb_extension.Makefile

# Build Rust library in release mode (called before cmake)
rust_release:
	cargo build --release

# Build Rust library in debug mode
rust_debug:
	cargo build

# Run Rust tests
rust_test:
	cargo test

# Override release to build Rust first
release: rust_release
	@$(MAKE) -f extension-ci-tools/makefiles/duckdb_extension.Makefile release EXT_NAME=$(EXT_NAME) EXT_CONFIG=$(EXT_CONFIG)

# Override debug to build Rust first
debug: rust_debug
	@$(MAKE) -f extension-ci-tools/makefiles/duckdb_extension.Makefile debug EXT_NAME=$(EXT_NAME) EXT_CONFIG=$(EXT_CONFIG)

# Clean everything
clean_all:
	rm -rf build
	cargo clean
