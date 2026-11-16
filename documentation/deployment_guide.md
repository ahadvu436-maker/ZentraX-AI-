# ZentraX AI — Deployment Guide

**Status:** Target deployment guidance. Validate all manifests against the actual application before use.

## 1. Deployment objectives

- repeatable deployments;
- explicit configuration;
- least-privilege runtime;
- health and readiness checks;
- controlled upgrades and rollback;
- encrypted communication;
- observable operation;
- recoverable state.

## 2. Environments

### Development
- synthetic data only;
- local credentials;
- debug mode restricted to loopback;
- no production secrets.

### Testing / staging
- production-like topology;
- isolated test accounts and datasets;
- security and integration tests;
- no uncontrolled customer data.

### Production
- hardened images;
- pinned dependencies;
- secrets manager;
- TLS;
- restricted network;
- backup and restore;
- monitoring and alerting;
- documented on-call and incident response.

## 3. Reference topology

A typical deployment may contain:
- reverse proxy / ingress;
- API service;
- background worker service;
- PostgreSQL;
- Redis/queue;
- object storage;
- local model runtime or approved external provider;
- telemetry collector and monitoring backend.

Only deploy components that the actual code uses. Avoid adding infrastructure simply because it appears in a reference architecture.

## 4. Container requirements

- Use a maintained minimal base image.
- Pin image tags or digests.
- Install only runtime dependencies.
- Run as a non-root user.
- Use read-only filesystem where practical.
- Drop Linux capabilities not required.
- Set CPU, memory, and process limits.
- Do not bake secrets into image layers.
- Scan images and dependencies.
- Generate an SBOM for release artifacts where possible.

## 5. Example Dockerfile pattern

This is a pattern, not a verified Dockerfile for the current repository. Adjust module path and dependencies.

```dockerfile
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

RUN useradd --create-home --uid 10001 appuser

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

USER appuser

EXPOSE 8000

CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT:-8000}"]
```

Use a Python version actually supported by the project's dependencies. The `main:app` import path must match the real application object.

## 6. Environment and secrets

- Inject secrets at runtime.
- Use separate secrets per environment.
- Rotate credentials after suspected exposure.
- Restrict who can read production secrets.
- Keep a documented recovery process for encryption keys.
- Never put real credentials in example files, screenshots, logs, or source control.

## 7. Network controls

- Publicly expose only the ingress endpoint.
- Keep databases, Redis, worker control ports, and model runtimes on private networks.
- Restrict outbound network access where feasible.
- Use TLS between services when the deployment threat model requires it.
- Apply explicit CORS allowlists.
- Use firewall/security-group rules and network segmentation.

## 8. Database operations

- Use migrations, not ad hoc schema changes.
- Back up before destructive migrations.
- Test migration forward and rollback strategy.
- Encrypt backups.
- Store backups separately from the primary server.
- Test restore on a schedule.
- Define recovery point objective (RPO) and recovery time objective (RTO).

## 9. Queue and worker operations

- Separate API and worker processes.
- Set concurrency based on measured CPU/GPU and memory capacity.
- Use task timeouts and retry limits.
- Monitor queue depth and oldest-task age.
- Provide a safe way to pause queues.
- Test worker termination during a task.
- Ensure a retry cannot duplicate irreversible side effects.

## 10. GPU deployment

- Confirm host driver and container runtime support.
- Pin model artifacts and verify checksums.
- Set per-model VRAM and concurrency limits.
- Avoid loading more models than available VRAM permits.
- Monitor utilization, temperature, memory, and throttling.
- Define model unload and worker restart behavior.
- Keep model downloads and licenses compliant.

## 11. Health checks and rollout

- Liveness: process is alive.
- Readiness: critical dependencies are usable.
- Startup: initialization has completed.
- Do not use a deep model inference as a frequent liveness probe.
- Use rolling/canary deployment only when state compatibility supports it.
- Verify migrations and compatibility before traffic shift.

## 12. Rollback

A release must define:
- previous image/version;
- database compatibility;
- migration rollback or forward-fix strategy;
- model artifact version;
- configuration version;
- verification commands;
- decision owner.

Never assume a database migration can always be reversed safely.

## 13. Monitoring and alerts

Monitor:
- request rate and latency;
- 4xx/5xx rates;
- task queue depth and age;
- worker failures;
- database connections and query latency;
- Redis health;
- CPU/RAM/disk;
- GPU utilization and VRAM;
- model latency and provider errors;
- authentication failures;
- privilege changes;
- emergency-stop activation.

Alerts must have an owner and a runbook.

## 14. Incident response

1. detect and classify;
2. contain affected credentials/services;
3. preserve relevant logs and evidence;
4. assess data and customer impact;
5. recover from known-good state;
6. rotate affected secrets;
7. notify stakeholders as required;
8. document root cause and corrective actions.

## 15. Deployment acceptance checklist

- [ ] Image built from reviewed commit.
- [ ] Dependencies pinned/scanned.
- [ ] Non-root runtime.
- [ ] Secrets injected securely.
- [ ] Public/private network boundaries verified.
- [ ] TLS and ingress settings verified.
- [ ] Database migrations applied.
- [ ] Backup and restore tested.
- [ ] Queue and worker behavior tested.
- [ ] GPU inference tested if required.
- [ ] Health/readiness checks work.
- [ ] Monitoring and alerts active.
- [ ] Rollback plan rehearsed.
- [ ] Security review completed.
