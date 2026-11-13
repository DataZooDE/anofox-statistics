PROJ_DIR := $(dir $(abspath $(lastword $(MAKEFILE_LIST))))

# Configuration of extension
EXT_NAME=anofox_statistics
EXT_CONFIG=${PROJ_DIR}extension_config.cmake

# Include the Makefile from extension-ci-tools
include extension-ci-tools/makefiles/duckdb_extension.Makefile

# Claude Code plugin: names resolve, every function is covered by a skill, and
# plugin.json / SKILL.md versions equal the newest CHANGELOG release
# (scripts/check_plugin.py; needs make release). bump-plugin sets the versions in
# the release commit.
.PHONY: check-plugin bump-plugin
check-plugin:
	python3 scripts/check_plugin.py

bump-plugin:
	./scripts/bump_plugin_version.sh

# Rust targets (for local development)
.PHONY: rust_release rust_debug rust_test

rust_release:
	cargo build --ignore-rust-version --release

rust_debug:
	cargo build --ignore-rust-version

rust_test:
	cargo test --ignore-rust-version

# Clean everything including Rust
clean_all:
	rm -rf build
	cargo clean
