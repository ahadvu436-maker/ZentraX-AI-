# ZentraX AI — Test Suite (19_tests)

This is a reusable pytest testing foundation. It does **not** claim that ZentraX is
secure, production-ready, or fully tested merely because these tests pass. The
suite includes real static checks plus optional integration/system/performance
checks that must be configured for the actual application.

## Install

From the repository root:

```bash
python -m pip install pytest
```

Optional tools can be added later (for example `pytest-cov`, `pytest-xdist`,
`bandit`, `pip-audit`, `hypothesis`, `httpx`, and `pytest-benchmark`) after
selecting versions compatible with the project.

## Run

```bash
python -m pytest 19_tests -v
```

Run by category:

```bash
python -m pytest 19_tests/unit -v
python -m pytest 19_tests/integration -v
python -m pytest 19_tests/security -v
python -m pytest 19_tests/system -v
python -m pytest 19_tests/performance -v
python -m pytest 19_tests/regression -v
```

## Configure actual ZentraX modules

Set import paths for modules that really exist in the repository:

```bash
# Linux/macOS
export ZENTRAX_IMPORT_TARGETS="backend.security,backend.model_management,backend.multimodal"

# PowerShell
$env:ZENTRAX_IMPORT_TARGETS="backend.security,backend.model_management,backend.multimodal"
```

Use Python import paths, not folder names guessed from the architecture. The
repository root must be importable and package names must match the actual code.

## Configure application tests

If the application provides a factory:

```bash
export ZENTRAX_APP_FACTORY="backend.main:create_app"
python -m pytest 19_tests/integration 19_tests/system -v
```

The factory must be a zero-argument callable returning the app. Do not set this
until that factory really exists.

## Deployed health check

```bash
export ZENTRAX_HEALTH_URL="https://your-test-host.example/health"
python -m pytest 19_tests/system -v
```

Use a staging/test deployment, not an unapproved production endpoint.

## Performance smoke test

Configure a safe zero-argument function:

```bash
export ZENTRAX_PERF_TARGET="backend.benchmark_targets:smoke_inference"
export ZENTRAX_PERF_MAX_SECONDS="10"
python -m pytest 19_tests/performance -v
```

This is a single-call smoke test, not a throughput, latency percentile, GPU
memory, concurrency, or stress benchmark. Those require workload-specific
benchmarks and controlled hardware.

## Security limitations

The static security checks are heuristics. They can produce false positives and
cannot replace threat modeling, dependency scanning, secrets scanning, SAST,
DAST, penetration testing, cryptographic review, or an independent audit.
Do not put real credentials in test fixtures or commit `.env` files.

## Important

- `__init__.py` marks conventional Python packages; it does not connect modules.
- Skipped tests mean a required target/configuration was not supplied; they are
  not a passing functional test.
- Add tests for each real ZentraX API, class, model adapter, security policy,
  queue, database repository, and control-system rule.
- Keep test data synthetic and remove secrets/real user data.
