# Makefile for libanostat
# Provides convenient build and test commands

.PHONY: all build test clean configure help

# Default target
all: build test

help:
	@echo "libanostat - Makefile targets:"
	@echo "  make configure  - Configure CMake build system"
	@echo "  make build      - Build the library and tests"
	@echo "  make test       - Run all unit tests"
	@echo "  make clean      - Remove build directory"
	@echo "  make all        - Configure, build, and test (default)"
	@echo ""
	@echo "Note: Requires Eigen3 and Catch2. Optional: nlohmann_json for validation tests"
	@echo ""
	@echo "Example with vcpkg:"
	@echo "  VCPKG_ROOT=/path/to/vcpkg make configure"

# Configure CMake build
configure:
	@if [ -n "$$VCPKG_ROOT" ]; then \
		echo "Configuring with vcpkg toolchain..."; \
		cmake -B build -S . \
			-DCMAKE_TOOLCHAIN_FILE=$$VCPKG_ROOT/scripts/buildsystems/vcpkg.cmake \
			-DBUILD_TESTING=ON; \
	else \
		echo "Configuring with system packages..."; \
		cmake -B build -S . -DBUILD_TESTING=ON; \
	fi

# Build library and tests
build:
	@if [ ! -d build ]; then \
		$(MAKE) configure; \
	fi
	cmake --build build -j

# Run tests using CTest
test: build
	@echo "Running libanostat unit tests..."
	ctest --test-dir build --output-on-failure

# Clean build artifacts
clean:
	rm -rf build

# Run a specific test by name
# Usage: make run-test TEST=distributions
run-test: build
	@if [ -z "$(TEST)" ]; then \
		echo "Error: TEST variable not set"; \
		echo "Usage: make run-test TEST=distributions"; \
		exit 1; \
	fi
	cd build && ctest -R "$(TEST)" --output-on-failure -V
