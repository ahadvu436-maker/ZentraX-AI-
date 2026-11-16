# ZentraX AI — System Design

**Status:** Target design; reconcile with actual code before implementation claims.

## 1. Scope

This document describes runtime components, request flow, state management, task lifecycle, module integration, and operational behavior.

## 2. Runtime components

### 2.1 API process
- Hosts HTTP endpoints.
- Parses and validates requests.
- Authenticates callers.
- Authorizes actions.
- Creates task records.
- Calls application services.
- Returns synchronous results or asynchronous task identifiers.

### 2.2 Application service layer
Contains use-case logic. It should not depend directly on HTTP request objects. Example services:
- `AuthenticationService`
- `AuthorizationService`
- `ModelRegistryService`
- `InferenceService`
- `TaskService`
- `EvaluationService`
- `AuditService`
- `ApprovalService`

These are interface names, not claims that matching classes already exist.

### 2.3 Worker processes
Workers execute long-running tasks such as document processing, multimodal inference, evaluation, or simulations. Workers must:
- receive a validated task envelope;
- re-check authorization for sensitive work;
- load only approved configuration;
- enforce resource/time budgets;
- emit lifecycle events;
- store results through controlled services;
- support cancellation where feasible.

### 2.4 Persistence
Recommended data stores:
- PostgreSQL (or another supported relational database) for durable records;
- Redis only for cache, short-lived coordination, and queue transport where configured;
- object storage for large files and artifacts;
- a vector store only if retrieval functionality requires it.

Do not treat Redis as the sole durable record of critical business state.

## 3. Request lifecycle

1. Client sends request over HTTPS.
2. Ingress applies TLS, size limits, rate limits, and request ID.
3. API validates content type and schema.
4. Authentication identifies the principal.
5. Authorization checks operation, tenant, resource, and policy.
6. Input safety checks run.
7. Application creates a request/task record.
8. Orchestrator selects synchronous or asynchronous execution.
9. Model adapter or worker performs the approved operation.
10. Output validation and safety checks run.
11. Result and audit metadata are persisted.
12. API returns result or task status.
13. Telemetry records latency, errors, and resource usage without leaking secrets.

## 4. Task state machine

Allowed states:
- `PENDING`
- `QUEUED`
- `RUNNING`
- `WAITING_FOR_APPROVAL`
- `SUCCEEDED`
- `FAILED`
- `CANCELLED`
- `TIMED_OUT`

Rules:
- Terminal states are `SUCCEEDED`, `FAILED`, `CANCELLED`, `TIMED_OUT`.
- A task cannot move from a terminal state back to `RUNNING`.
- State transitions must be validated centrally.
- Retries should create attempt records or increment a controlled attempt counter.
- Duplicate client submissions should be handled with an idempotency key where supported.
- A task requiring human approval must not execute the protected action before approval.

## 5. Module integration pattern

Each module exposes a narrow interface and a typed configuration model. The composition root wires concrete implementations.

Recommended pattern:
- `ports/`: interfaces or protocols;
- `adapters/`: database, provider, queue, hardware implementations;
- `services/`: business use cases;
- `schemas/`: validated input/output contracts;
- `api/`: transport layer.

Avoid circular imports. Avoid importing `main.py` from service modules. Avoid import-time side effects such as opening database connections or starting workers.

## 6. Model selection and inference

The model registry should record:
- model identifier;
- provider/runtime;
- version or immutable revision;
- modality support;
- context and output limits;
- hardware requirements;
- license/usage constraints;
- safety configuration;
- enabled/disabled status.

Selection rules:
1. Resolve requested model alias through the registry.
2. Confirm model is enabled and permitted for the caller/tenant.
3. Check input modality and size.
4. Check provider policy and data classification.
5. Check available capacity and resource budget.
6. Dispatch through the provider adapter.
7. Record actual provider/model/version used.
8. Validate output and return a normalized response.

Do not silently send confidential data to an external provider. The provider route must be explicit and policy-controlled.

## 7. Multi-agent execution

Agents should be treated as bounded workers, not privileged administrators.

Every agent task must have:
- a defined objective;
- an allowed tool list;
- a permission scope;
- a maximum step count;
- a deadline;
- a token/compute budget;
- an output schema;
- an audit/correlation ID;
- cancellation behavior.

Agents must not:
- grant themselves permissions;
- modify governance policy;
- disable security monitoring;
- access secrets outside their scope;
- execute arbitrary shell commands without a sandbox and approval policy;
- treat untrusted retrieved content as instructions.

Delegation must preserve the original user's authorization boundary.

## 8. Human approval and control plane

Actions requiring approval should be explicitly classified, for example:
- external communications;
- financial or contractual actions;
- destructive data operations;
- changes to access policy;
- deployment to production;
- code changes in protected branches;
- actions with irreversible external impact.

Approval record should include:
- requester;
- approver;
- action summary;
- target resources;
- risk classification;
- policy version;
- expiry;
- decision and timestamp.

Emergency stop should:
1. prevent new autonomous work;
2. signal active workers to stop safely;
3. revoke or suspend task credentials where appropriate;
4. preserve audit evidence;
5. require an authorized operator to restore service.

## 9. Error model

Use stable machine-readable error codes, for example:
- `VALIDATION_ERROR`
- `AUTHENTICATION_REQUIRED`
- `PERMISSION_DENIED`
- `RESOURCE_NOT_FOUND`
- `RATE_LIMITED`
- `MODEL_UNAVAILABLE`
- `TASK_TIMEOUT`
- `APPROVAL_REQUIRED`
- `DEPENDENCY_UNAVAILABLE`
- `INTERNAL_ERROR`

Never return stack traces, secrets, SQL details, filesystem paths, or provider credentials to clients.

## 10. Idempotency, retries and timeouts

- Define idempotency behavior for task creation and other non-read operations.
- Use bounded exponential backoff with jitter for retryable provider/network errors.
- Do not retry validation, authorization, or policy denials.
- Set connect, read, total, and worker execution timeouts.
- Use a dead-letter strategy for repeatedly failing queued tasks.
- Ensure retry behavior cannot create duplicate external side effects.

## 11. Data model baseline

Suggested entities:
- `users`
- `tenants`
- `roles`
- `permissions`
- `api_credentials` (store hashes or secure references, never raw secrets)
- `models`
- `model_versions`
- `tasks`
- `task_attempts`
- `approvals`
- `audit_events`
- `provider_configs` (secret references only)
- `evaluation_runs`
- `artifacts`

Actual schema must be created through versioned migrations and reviewed for tenant isolation.

## 12. Observability design

Every request/task should carry:
- `request_id`
- `correlation_id`
- `task_id` when asynchronous
- `tenant_id` where applicable
- `actor_id` or privacy-safe principal reference
- `module`
- `operation`
- `outcome`

Metrics should avoid high-cardinality labels such as raw user IDs or prompts.

## 13. Startup and shutdown

Startup:
1. load and validate configuration;
2. initialize structured logging;
3. validate required dependencies;
4. initialize database connection pool;
5. initialize queue/provider adapters;
6. register routes and health checks;
7. start serving traffic only after critical dependencies are ready.

Shutdown:
1. stop accepting new work;
2. drain or safely requeue in-flight work;
3. close provider and database clients;
4. flush telemetry;
5. exit within the configured grace period.

## 14. Design validation checklist

- [ ] No circular imports.
- [ ] No secrets committed.
- [ ] API schema validation enabled.
- [ ] Authorization enforced at service/resource level.
- [ ] Task transitions validated.
- [ ] Idempotency documented.
- [ ] Worker retry policy bounded.
- [ ] Provider routing explicit.
- [ ] Approval gates tested.
- [ ] Emergency stop tested.
- [ ] Audit events generated.
- [ ] Backups and restore tested.
