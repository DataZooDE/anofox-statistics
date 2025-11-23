# Makefile for libanostat
# Provides convenient build and test commands

.PHONY: all build test clean configure help coverage

# Default target
all: build test

help:
	@echo "libanostat - Makefile targets:"
	@echo "  make configure  - Configure CMake build system"
	@echo "  make build      - Build the library and tests"
	@echo "  make test       - Run all unit tests"
	@echo "  make coverage   - Run tests with coverage analysis (requires lcov)"
	@echo "  make clean      - Remove build directory"
	@echo "  make all        - Configure, build, and test (default)"
	@echo ""
	@echo "Note: Requires Eigen3 and Catch2. Optional: nlohmann_json for validation tests"
	@echo "      Coverage requires lcov: sudo pacman -S lcov"
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

# Run tests with coverage analysis
# Requires: lcov (install via: sudo pacman -S lcov)
# Outputs: coverage_html/index.html (HTML report) and coverage_filtered.info (lcov data)
coverage:
	@echo "=== Running coverage analysis for libanostat ==="
	@which lcov > /dev/null 2>&1 || (echo "Error: lcov not found. Install with: sudo pacman -S lcov" && exit 1)
	@which genhtml > /dev/null 2>&1 || (echo "Error: genhtml not found. Install with: sudo pacman -S lcov" && exit 1)
	@echo "Cleaning previous coverage data..."
	@rm -rf build/coverage coverage_html *.info
	@echo "Configuring with coverage flags..."
	@if [ -n "$$VCPKG_ROOT" ]; then \
		cmake -B build/coverage -S . \
			-DCMAKE_TOOLCHAIN_FILE=$$VCPKG_ROOT/scripts/buildsystems/vcpkg.cmake \
			-DCMAKE_BUILD_TYPE=Debug \
			-DCMAKE_CXX_FLAGS="--coverage -fno-inline -fno-elide-constructors -O0" \
			-DBUILD_TESTING=ON; \
	else \
		cmake -B build/coverage -S . \
			-DCMAKE_BUILD_TYPE=Debug \
			-DCMAKE_CXX_FLAGS="--coverage -fno-inline -fno-elide-constructors -O0" \
			-DBUILD_TESTING=ON; \
	fi
	@echo "Building tests with coverage instrumentation..."
	@cmake --build build/coverage -j
	@echo "Initializing coverage baseline..."
	@lcov --config-file .lcovrc --zerocounters --directory build/coverage
	@lcov --config-file .lcovrc --capture --initial --directory build/coverage --output-file coverage_base.info
	@echo "Running tests..."
	@cd build/coverage && ctest --output-on-failure
	@echo "Capturing coverage data..."
	@lcov --config-file .lcovrc --capture --directory build/coverage --output-file coverage_test.info
	@echo "Combining baseline and test coverage..."
	@lcov --config-file .lcovrc --add-tracefile coverage_base.info --add-tracefile coverage_test.info --output-file coverage_total.info
	@echo "Filtering coverage data (removing test code, external libraries)..."
	@lcov --config-file .lcovrc --remove coverage_total.info \
		'*/test/*' '*/tests/*' '*/catch2/*' '*/Catch2/*' \
		'*/eigen3/*' '*/Eigen/*' '/usr/*' '*/vcpkg_installed/*' \
		--output-file coverage_filtered.info
	@echo "Generating HTML report..."
	@genhtml coverage_filtered.info --output-directory coverage_html \
		--title "libanostat Coverage Report" \
		--legend --show-details
	@echo ""
	@echo "=== Coverage Report Summary ==="
	@lcov --config-file .lcovrc --summary coverage_filtered.info
	@echo ""
	@echo "Coverage report generated successfully!"
	@echo "  HTML report: coverage_html/index.html"
	@echo "  Coverage data: coverage_filtered.info"
	@echo ""
	@echo "To view the report, open: file://$(shell pwd)/coverage_html/index.html"
