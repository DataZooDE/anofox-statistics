# External Integrations

**Analysis Date:** 2026-08-11

## APIs & External Services

**PostHog Telemetry:**
- PostHog analytics - Sends anonymized usage telemetry to PostHog SaaS
  - SDK/Client: Custom C++ implementation via `posthog-telemetry/` (vendored)
  - HTTP: `httplib.hpp` (header-only C++ HTTP client with OpenSSL)
  - TLS: OpenSSL 1.1+ (via `find_package(OpenSSL REQUIRED)`)
  - Endpoint: PostHog cloud API (https://posthog.com)
  - Status: **Disabled on MinGW builds** where OpenSSL unavailable (cmake flag `ANOFOX_TELEMETRY_ENABLED`)
  - Location: `CMakeLists.txt` lines 84–106, `posthog-telemetry/src/telemetry.cpp`

**GitHub:**
- Repository hosting: https://github.com/DataZooDE/anofox-statistics
- CI/CD workflows: GitHub Actions (`.github/workflows/`)
- Dependency sourcing: Custom fork `https://github.com/DataZooDE/argmin.git` (branch `fix/stable-rust-compat`)

## Data Storage

**Databases:**
- DuckDB (embedded) - No external database connection
  - Architecture: Extension runs in-process within DuckDB process memory
  - Data handled entirely via DuckDB's query execution engine
  - No separate database service required

**File Storage:**
- Local filesystem only - Extension reads/writes via DuckDB's I/O subsystem
- No cloud storage integrations (S3, GCS, Azure Blob Storage)

**Caching:**
- None - Stateless extension; all computation is request-scoped

## Authentication & Identity

**Auth Provider:**
- None - Extension requires no authentication
- PostHog telemetry: API key hardcoded or injected at build time (machine ID for anonymization)

**Machine Identification (Telemetry):**
- Platform detection: Compile-time (`posthog-telemetry/src/telemetry.cpp` lines 64–80)
- OS/Architecture split: `ComputeOs()` / `ComputeArch()` functions
- Machine ID collection:
  - **Linux:** `/etc/machine-id` or systemd-dbus
  - **macOS:** IORegistry queries (requires `IOKit` framework)
  - **Windows:** WMI queries via `iphlpapi.h`

## Monitoring & Observability

**Error Tracking:**
- None - DuckDB handles error reporting

**Logs:**
- Console output via C++ `std::cout` / `std::cerr` (DuckDB captures and routes)
- Telemetry sends error summaries to PostHog for aggregation
- Location: `posthog-telemetry/src/telemetry.cpp` (error batching queue)

**Telemetry Events Tracked:**
- Extension load/unload lifecycle
- Aggregate function invocations
- Table function invocations
- Execution duration (milliseconds)
- Platform/architecture metadata
- DuckDB version compatibility
- Opt-out: Set `POSTHOG_TELEMETRY_DISABLED` compile flag (automatic on MinGW)

## CI/CD & Deployment

**Hosting:**
- GitHub Actions - Build and test orchestration
- GitHub Releases - Extension binary distribution
- Reusable workflows from `duckdb/extension-ci-tools` (v1.5-variegata, v1.4-andium)

**CI Pipeline:**
- **Build matrix:** Linux (amd64, arm64), macOS (amd64, arm64), WASM
- **Excluded:** Windows (`windows_amd64`, `windows_amd64_mingw`) — vcpkg OpenSSL 404 on MinGW; MSVC fmt flakiness
- **Workflows:**
  - `.github/workflows/MainDistributionPipeline.yml` - Multi-version matrix (v1.5.5 stable, v1.4.5 LTS)
  - `.github/workflows/_extension_deploy.yml` - Binary release to GitHub + DuckDB registry
- **Deploy triggers:**
  - Automatic on merge to `main` branch
  - Automatic on git tags matching `v*` (semantic versioning)
  - Manual via `workflow_dispatch`

**Binary Distribution:**
- DuckDB extension registry - Published via POST to registry endpoint
- GitHub releases - Attached as `.so`, `.dll`, `.dylib` artifacts
- OIDC authentication - GitHub Actions uses OIDC token for AWS credential assumption (deploy job)

**Rust Dependency Sourcing:**
- Upstream dependencies from crates.io registry (default)
- Custom patched fork of `argmin` and `argmin-math` from DataZooDE GitHub (branch: `fix/stable-rust-compat`)
  - Reason: Standard argmin requires nightly Rust; custom fork adapts to stable channel
  - Location: `Cargo.toml` lines 30–32

## Environment Configuration

**Required env vars:**
- None at runtime - Extension is stateless
- Build-time vars:
  - `Rust_CARGO_TARGET` - Set by CMake based on detected OS/arch (internal)
  - `ANOFOX_TELEMETRY_ENABLED` - Set by CMake; disabled on MinGW (internal)

**Optional env vars:**
- `POSTHOG_API_KEY` - Can override hardcoded PostHog API key (if telemetry is enabled)
- `DUCKDB_EXTENSION_DIRECTORY` - DuckDB runtime; not set by this extension

**Secrets location:**
- None in repo
- `.env` files present (if any) are git-ignored
- PostHog API key: Embedded at compile time (hardcoded in telemetry.cpp or CI secret injection)

## Webhooks & Callbacks

**Incoming:**
- None - Extension does not listen for incoming HTTP

**Outgoing:**
- PostHog analytics endpoint (HTTPS POST)
  - Batched events queue (`posthog-telemetry/src/telemetry.cpp` lines 600+)
  - Background thread sends async (non-blocking to DuckDB execution)
  - Endpoint: `https://us.posthog.com/capture` or configurable via API key subdomain

---

*Integration audit: 2026-08-11*
