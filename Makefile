PROJ_DIR := $(dir $(abspath $(lastword $(MAKEFILE_LIST))))

# Configuration of extension
EXT_NAME=anofox_statistics
EXT_CONFIG=${PROJ_DIR}extension_config.cmake

# Include the Makefile from extension-ci-tools
include extension-ci-tools/makefiles/duckdb_extension.Makefile

# Rust targets (for local development)
.PHONY: rust_release rust_debug rust_test

rust_release:
	cargo build --release

rust_debug:
	cargo build

rust_test:
	cargo test

# Clean everything including Rust
clean_all:
	rm -rf build
	cargo clean
