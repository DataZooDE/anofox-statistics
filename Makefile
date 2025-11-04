PROJ_DIR := $(dir $(abspath $(lastword $(MAKEFILE_LIST))))

# Configuration of extension
EXT_NAME=anofox_statistics
EXT_CONFIG=${PROJ_DIR}extension_config.cmake

# Include the Makefile from extension-ci-tools
include extension-ci-tools/makefiles/duckdb_extension.Makefile

# Documentation targets
.PHONY: docs test-docs clean-docs install-hooks lint-docs

# Build documentation from templates
docs:
	@bash scripts/build_docs.sh

# Test SQL examples
test-docs:
	@bash scripts/test_sql_examples.sh

# Lint markdown files
lint-docs:
	@echo "📝 Linting markdown files..."
	@markdownlint 'guides/**/*.md' --ignore node_modules

# Clean generated documentation
clean-docs:
	@echo "🧹 Cleaning generated documentation..."
	@find guides -name "*.md" -not -name "*.md.in" -type f -exec rm -f {} \;
	@echo "✅ Cleaned"

# Install git hooks
install-hooks:
	@bash scripts/install_hooks.sh

# Validation targets
.PHONY: generate-test-data test-validation clean-test-data

# Generate test data (requires R with jsonlite and glmnet packages)
generate-test-data:
	@echo "🔄 Regenerating test data..."
	@bash validation/generate_all_data.sh

# Run validation tests (SQL only, no R needed)
test-validation:
	@echo "🧪 Running validation tests..."
	@bash scripts/test_sql_validation.sh

# Clean generated test data
clean-test-data:
	@echo "🧹 Cleaning test data..."
	@rm -rf test/data/*/input/* test/data/*/expected/*
	@echo "✅ Cleaned test data (metadata and READMEs preserved)"