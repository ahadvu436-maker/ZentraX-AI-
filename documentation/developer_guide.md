# ZentraX AI — Developer Guide

## 1. Purpose

This guide defines how contributors should understand, modify, test, and review the codebase. It must be updated to match the actual repository conventions.

## 2. Repository orientation

Start by identifying:
- application entry point;
- package roots;
- dependency files and lockfiles;
- configuration loader;
- API router registration;
- service layer;
- database models and migrations;
- worker/queue entry point;
- model adapters;
- test directories;
- deployment files;
- documentation ownership.

Do not infer implementation from folder names alone.

## 3. Python package rules

- Use `__init__.py` for regular packages where appropriate.
- Keep package initialization lightweight.
- Do not import every submodule automatically unless there is a clear reason.
- Avoid side effects in `__init__.py`.
- Avoid circular imports.
- Do not import `main.py` from service modules.
- Prefer dependency injection over hidden global singletons.
- Use type hints for public interfaces.
- Keep public API stable and documented.

An empty `__init__.py` is valid. It marks a regular package but does not automatically connect its files to the application.

## 4. Code organization

Recommended responsibility split:
- `api/`: HTTP transport and request/response mapping;
- `schemas/`: validated input/output models;
- `services/`: use-case logic;
- `domain/`: core business rules and entities;
- `ports/`: interfaces/protocols;
- `adapters/`: database, provider, queue, hardware integrations;
- `core/`: configuration, logging, errors;
- `tests/`: unit, integration, security, end-to-end tests.

Adapt this structure to the current project rather than duplicating folders blindly.

## 5. Configuration

- Validate configuration at startup.
- Use typed settings.
- Fail startup when required settings are missing or malformed.
- Never include secrets in default values.
- Distinguish development, testing, staging, and production.
- Do not allow debug mode in production.
- Document every variable and whether it is secret.

## 6. Error handling

- Raise domain-specific exceptions inside services.
- Map exceptions to stable API error responses at the API boundary.
- Do not swallow exceptions silently.
- Do not expose stack traces to users.
- Include correlation IDs in logs.
- Preserve original exception context for internal diagnostics.

## 7. Logging

Use structured logs. Include timestamp, severity, service, module, operation, correlation ID, and outcome. Redact secrets and sensitive content. Avoid logging full prompts and files unless a documented, consented diagnostic mode requires it.

## 8. Testing standards

### Unit tests
Test a single function/class with dependencies mocked or controlled.

### Integration tests
Test interactions with actual supported dependencies, such as a temporary database or queue.

### Contract tests
Verify that adapters and modules satisfy their declared interfaces.

### Security tests
Verify denial cases, tenant isolation, invalid tokens, privilege boundaries, and unsafe inputs.

### End-to-end tests
Exercise a realistic user journey from API request through task/model processing to response.

Tests must be deterministic where possible and must clean up their data.

## 9. Pull request checklist

- [ ] Scope and reason are clear.
- [ ] Public interfaces documented.
- [ ] Input validation added.
- [ ] Authorization considered.
- [ ] Secrets not added.
- [ ] Error behavior defined.
- [ ] Tests added/updated.
- [ ] Lint/type checks pass.
- [ ] Dependency changes justified and pinned.
- [ ] Logging avoids sensitive data.
- [ ] Migration/rollback considered.
- [ ] Documentation updated.
- [ ] Security impact reviewed.

## 10. Dependency management

- Pin versions for releases.
- Review licenses and maintenance status.
- Avoid unnecessary dependencies.
- Run vulnerability scans.
- Record runtime/toolchain versions.
- Test upgrades in a separate branch.

## 11. Model adapter development

Each model adapter should implement a common contract:
- capability metadata;
- health check;
- request validation;
- invocation;
- normalized response;
- timeout/cancellation behavior;
- usage/latency metadata;
- error mapping;
- privacy/provider metadata.

Do not let application services depend on provider-specific response formats.

## 12. Hardware adapter development

Hardware adapters should:
- detect available device/runtime;
- report capability and memory;
- expose safe allocation limits;
- handle device unavailable/out-of-memory errors;
- support CPU fallback only when explicitly configured;
- emit resource metrics;
- never assume a GPU is present.

## 13. Database changes

- Use versioned migrations.
- Add indexes based on query patterns and measurement.
- Avoid destructive changes without a migration plan.
- Test upgrade from the previous supported schema.
- Test restore and migration compatibility.

## 14. Release process

1. create release branch/tag;
2. pin dependencies and model versions;
3. run test suites;
4. scan secrets/dependencies/images;
5. build artifact;
6. produce release notes and SBOM where possible;
7. deploy to staging;
8. run smoke and regression tests;
9. obtain release approval;
10. deploy and monitor;
11. retain rollback artifact.

## 15. Contribution and review

All changes affecting authentication, authorization, encryption, autonomy, tool execution, data retention, or production deployment require additional security review.

## 16. Definition of done

A task is done when code, tests, documentation, operational behavior, and review evidence are complete—not when files have merely been created.
