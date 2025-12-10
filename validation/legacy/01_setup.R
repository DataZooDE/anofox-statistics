# Validation Setup Script
# This script installs required packages and sets up the validation environment

cat("=== Anofox Statistics Extension Validation Setup ===\n\n")

# Set up local library path (writable)
local_lib <- file.path(getwd(), "R_libs")
if (!dir.exists(local_lib)) {
  dir.create(local_lib, recursive = TRUE)
}
.libPaths(c(local_lib, .libPaths()))
cat("Using R library path:", local_lib, "\n\n")

# Required packages
required_packages <- c(
  "DBI",        # Database interface
  "duckdb",     # DuckDB R interface
  "glmnet",     # Ridge regression
  "car",        # VIF calculations
  "tseries",    # Jarque-Bera test
  "zoo",        # Rolling/expanding windows
  "dplyr",      # Data manipulation
  "testthat",   # Testing framework
  "knitr",      # Report generation
  "rmarkdown"   # Report formatting
)

# Install missing packages
install_if_missing <- function(pkg) {
  if (!require(pkg, character.only = TRUE, quietly = TRUE)) {
    cat("Installing package:", pkg, "\n")
    install.packages(pkg, repos = "https://cloud.r-project.org/")
    library(pkg, character.only = TRUE)
  } else {
    cat("✓ Package", pkg, "already installed\n")
  }
}

cat("Checking required packages...\n")
invisible(sapply(required_packages, install_if_missing))

cat("\n=== Setup Complete ===\n")
cat("R version:", R.version.string, "\n")
cat("Platform:", R.version$platform, "\n")

# Print loaded package versions
cat("\nPackage versions:\n")
for (pkg in required_packages) {
  version <- packageVersion(pkg)
  cat("  ", pkg, ":", as.character(version), "\n")
}

cat("\nYou can now run the validation tests.\n")
cat("Next step: Run validation/R/02_test_ols.R\n")
