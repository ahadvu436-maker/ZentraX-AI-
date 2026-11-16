# ZentraX AI — Installation Guide

**Status:** Baseline guide. Replace example commands and dependencies with the verified project-specific values.

## 1. Supported installation modes

1. Local development on a workstation.
2. CPU-only integration environment.
3. GPU-enabled inference environment.
4. Containerized staging/production deployment.

Do not assume that every module requires a GPU. API, security, database, queue, orchestration, and most tests can often run on CPU. GPU requirements depend on the selected model, quantization, context length, batch size, and concurrency.

## 2. Prerequisites

- Supported operating system (Linux is recommended for server deployment).
- Python version matching the project's pinned dependencies.
- Git.
- A virtual environment tool.
- Docker Engine and Docker Compose plugin if using containers.
- PostgreSQL if required by the configured application.
- Redis if queue/cache functionality is enabled.
- NVIDIA driver and compatible CUDA/runtime stack for NVIDIA GPU inference.
- Adequate disk space for model weights, datasets, logs, and artifacts.

Record exact versions in the release manifest. Do not guess Python, CUDA, or driver versions.

## 3. Obtain source

```bash
git clone https://github.com/ahadvu436-maker/ZentraX-AI-.git
cd ZentraX-AI-
git status
```

Use a reviewed commit hash for reproducible tests and releases. Do not deploy directly from an unreviewed moving branch in production.

## 4. Create a virtual environment

Linux/macOS:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

Windows PowerShell:

```powershell
py -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

Install only dependencies declared by the repository:

```bash
pip install -r requirements.txt
```

If the repository does not yet contain a reliable dependency lock/requirements file, create one from verified imports and tested versions before claiming reproducibility.

## 5. Environment configuration

Create a local environment file from a committed example template if one exists:

```bash
cp .env.example .env
```

If `.env.example` does not exist, do not invent production secrets. Create a documented template containing variable names and safe placeholders, then obtain real secrets through a secure secret manager or local environment.

Typical configuration categories:
- application environment;
- bind host and port;
- database connection string;
- Redis connection string;
- signing/encryption key references;
- model provider selection;
- model cache/artifact directory;
- logging level;
- CORS allowlist;
- allowed upload size;
- request and task timeouts.

Never commit `.env`, API keys, private keys, database passwords, or production credentials.

## 6. Package and import structure

An empty `__init__.py` marks a regular Python package. It does not automatically import every module and does not connect every folder to `main.py`.

After adding package markers:
- use absolute imports from a stable package root;
- avoid importing the entry-point module from application modules;
- avoid circular imports;
- keep import-time side effects minimal;
- add tests that import every intended public module.

Example:

```python
from backend.security.authentication import AuthenticationService
```

The exact import path must match the actual repository structure.

## 7. Database setup

If PostgreSQL is used:
1. create a dedicated database and least-privilege application role;
2. configure the connection through a secret reference;
3. apply versioned migrations;
4. verify schema version;
5. run database integration tests;
6. test backup and restore.

Do not run application traffic against an un-migrated schema.

## 8. Redis and worker setup

If Redis/Celery or another queue is used:
- configure authentication and network restrictions;
- do not expose Redis publicly;
- define queue names and worker concurrency;
- define task timeouts and retry budgets;
- configure dead-letter or failed-task inspection;
- test duplicate delivery and worker crash behavior.

## 9. GPU readiness

Before installing or downloading large models:
1. identify the GPU model;
2. record VRAM;
3. record driver and CUDA compatibility;
4. run a minimal framework device check;
5. measure free VRAM at idle;
6. test one approved model at a small workload;
7. record peak VRAM, latency, throughput, and errors.

Do not select a GPU server solely by its advertised model name. Match VRAM and memory bandwidth to the actual model and workload.

## 10. Start the application

Use the actual ASGI/WSGI import path in the repository. Example only:

```bash
uvicorn main:app --host 127.0.0.1 --port 8000
```

For a container or hosted platform, bind to `0.0.0.0` and use the platform-provided port variable. Shell variable expansion differs between shell-form and exec-form Docker commands.

## 11. Initial verification

- [ ] Python environment activates.
- [ ] Dependencies install from pinned requirements.
- [ ] Configuration validation succeeds.
- [ ] Application imports successfully.
- [ ] Liveness endpoint responds.
- [ ] Readiness endpoint reports dependency state.
- [ ] Database migrations succeed.
- [ ] Queue/worker health is verified if enabled.
- [ ] Model adapter can run a small approved test.
- [ ] Logs do not contain secrets.
- [ ] Shutdown completes cleanly.

## 12. Troubleshooting

### Import error
Check package root, `__init__.py`, working directory, and import paths. Do not solve circular imports by importing `main.py` into lower-level modules.

### Port error
Ensure the process receives an integer port. If using environment variables, ensure the shell expands them. Do not pass a literal string such as `${PORT:-8000}` to an argument parser that expects an integer.

### Database unavailable
Check host, port, credentials, network policy, TLS requirements, and migration state. Do not print credentials while debugging.

### GPU unavailable
Check driver/runtime compatibility, container GPU passthrough, device visibility, VRAM, and model runtime logs.

### Model out of memory
Reduce model size/quantization/context/batch/concurrency or use a GPU with more VRAM. Record the exact tested configuration.

## 13. Reproducibility

For every tested release retain:
- commit hash;
- Python and dependency lock versions;
- OS/container image digest;
- GPU and driver details;
- model name/version/checksum/license;
- environment variable names (not secret values);
- migration version;
- test commands and results.
