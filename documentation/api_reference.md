# ZentraX AI — API Reference

**Status:** Proposed API contract. These endpoints are not asserted to exist in the current codebase. Confirm actual routes in the application before publishing this as an implemented API.

## 1. API conventions

- Base path: `/api/v1` (proposed).
- Transport: HTTPS in deployed environments.
- Content type: `application/json`, except explicitly documented file uploads.
- Encoding: UTF-8.
- Authentication: bearer access token or configured identity integration.
- Authorization: evaluated for every protected operation.
- Versioning: breaking changes require a new API version or a documented migration period.
- Time: ISO 8601 UTC timestamps.
- IDs: opaque identifiers; do not expose sequential database IDs where avoidable.

## 2. Standard response envelope

Success example:

```json
{
  "data": {},
  "request_id": "req_example",
  "timestamp": "2026-01-01T12:00:00Z"
}
```

Error example:

```json
{
  "error": {
    "code": "VALIDATION_ERROR",
    "message": "The request is invalid.",
    "details": []
  },
  "request_id": "req_example",
  "timestamp": "2026-01-01T12:00:00Z"
}
```

Do not expose internal stack traces or secrets in `details`.

## 3. Proposed endpoints

### 3.1 Health and readiness

`GET /health/live`
- Purpose: process liveness.
- Must not expose configuration, credentials, internal hostnames, or sensitive dependency details.
- Returns a minimal status.

`GET /health/ready`
- Purpose: whether the instance can accept work.
- May check critical dependencies with short timeouts.
- Return unavailable when critical dependencies are not ready.

### 3.2 Authentication

`POST /api/v1/auth/login`
- Proposed only if local username/password authentication is supported.
- Validate credentials using a maintained password hashing implementation.
- Apply rate limiting and account lockout/risk controls.
- Return short-lived access token and secure refresh mechanism if implemented.
- Never log passwords or tokens.

`POST /api/v1/auth/refresh`
- Rotate refresh tokens where supported.
- Detect reuse and revoke affected sessions according to policy.

`POST /api/v1/auth/logout`
- Revoke the current session/refresh token where supported.

If OIDC is used, prefer a standards-based provider flow rather than implementing a custom identity protocol.

### 3.3 Model registry

`GET /api/v1/models`
- Lists models available to the caller.
- Return only models permitted by tenant and policy.

`GET /api/v1/models/{model_id}`
- Returns safe model metadata, not provider secrets.

`POST /api/v1/models/{model_id}/enable`
- Administrative operation.
- Requires explicit permission and audit event.

### 3.4 Inference

`POST /api/v1/inference`
Proposed request:

```json
{
  "model": "configured-model-alias",
  "input": {
    "type": "text",
    "content": "Example prompt"
  },
  "options": {
    "max_output_tokens": 512,
    "temperature": 0.2
  },
  "mode": "sync"
}
```

Rules:
- Validate model alias against registry.
- Enforce caller permissions, input limits, and provider policy.
- Do not accept arbitrary provider URLs from clients.
- Treat all user input as untrusted.
- Enforce output limits and deadlines.
- Record model/provider/version actually used.

Possible synchronous response:

```json
{
  "data": {
    "output": {
      "type": "text",
      "content": "Example response"
    },
    "model": "configured-model-alias",
    "usage": {
      "input_tokens": null,
      "output_tokens": null
    }
  },
  "request_id": "req_example",
  "timestamp": "2026-01-01T12:00:00Z"
}
```

Usage values may be null if the provider does not supply them. Do not fabricate token counts.

### 3.5 Asynchronous tasks

`POST /api/v1/tasks`
- Creates a task when the operation is long-running.
- Supports an idempotency key where appropriate.
- Returns `202 Accepted` only after the task is durably accepted.

`GET /api/v1/tasks/{task_id}`
- Returns state, progress if available, timestamps, and safe error information.

`POST /api/v1/tasks/{task_id}/cancel`
- Requires ownership or administrative permission.
- Cancellation is best-effort if execution has already crossed an irreversible boundary.

### 3.6 Approvals and governance

`GET /api/v1/approvals`
- Lists approvals visible to the caller.

`POST /api/v1/approvals/{approval_id}/decision`
- Requires approver permission.
- Request includes `decision` (`approve` or `reject`) and optional reason.
- Record approver, policy version, timestamp, and decision.
- Never allow the requesting agent to approve its own protected action unless a documented policy explicitly permits it.

### 3.7 Evaluation

`POST /api/v1/evaluations`
- Starts an authorized evaluation run against an approved dataset/model configuration.
- Must not expose private evaluation data to unauthorized callers.

`GET /api/v1/evaluations/{evaluation_id}`
- Returns status and summary.

## 4. HTTP status guidance

- `200`: successful read or synchronous operation.
- `201`: resource created.
- `202`: asynchronous work accepted.
- `204`: successful operation with no response body.
- `400`: malformed request.
- `401`: missing/invalid authentication.
- `403`: authenticated but not permitted.
- `404`: resource not found or intentionally concealed.
- `409`: state conflict or idempotency conflict.
- `413`: payload too large.
- `422`: schema/semantic validation error.
- `429`: rate limit exceeded.
- `500`: unexpected internal failure.
- `502/503/504`: upstream/dependency unavailable or timeout.

## 5. API security requirements

- Authenticate all non-public endpoints.
- Authorize every operation and resource.
- Validate schemas and content types.
- Enforce request size and rate limits.
- Use TLS.
- Apply CSRF protection when cookie-based browser sessions are used.
- Configure CORS with explicit allowed origins; never use wildcard credentials.
- Redact secrets and sensitive content from logs.
- Avoid user-controlled file paths and URLs.
- Scan uploaded files where applicable.
- Use pagination and bounded result sizes.
- Publish OpenAPI only after routes and schemas are verified.

## 6. API compatibility and deprecation

For each endpoint document:
- owner;
- stability level;
- request/response schema;
- permission;
- rate limit;
- timeout;
- side effects;
- audit event;
- error codes;
- deprecation date and replacement if applicable.

## 7. Implementation verification

Before marking this reference as final:
1. inspect actual FastAPI/Flask route registration;
2. compare real paths and methods with this proposal;
3. export actual OpenAPI schema;
4. run endpoint tests;
5. verify auth and permission failures;
6. verify response redaction;
7. update this document to match actual behavior.
