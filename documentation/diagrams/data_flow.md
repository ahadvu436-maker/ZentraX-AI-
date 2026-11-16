# Data Flow Diagram

This diagram describes a generic request path. Adapt it to the actual implemented services and data stores.

```mermaid
sequenceDiagram
    autonumber
    actor User
    participant Ingress
    participant API
    participant Auth as AuthN/AuthZ
    participant Policy
    participant Orchestrator
    participant Queue
    participant Worker
    participant Model as Model Adapter
    participant Store as Database/Object Store
    participant Audit

    User->>Ingress: HTTPS request
    Ingress->>API: Forward request + request ID
    API->>API: Validate schema and size
    API->>Auth: Authenticate principal
    Auth-->>API: Principal / reject
    API->>Policy: Authorize operation and resource
    Policy-->>API: Allow / deny / approval required
    API->>Audit: Record security decision
    alt Denied
        API-->>User: 401/403 response
    else Approval required
        API->>Store: Create approval request
        API-->>User: Approval pending
    else Allowed
        API->>Orchestrator: Create task
        Orchestrator->>Store: Persist task
        Orchestrator->>Queue: Enqueue task
        API-->>User: 202 + task ID
        Queue->>Worker: Deliver task
        Worker->>Policy: Re-check scoped permission
        Policy-->>Worker: Permit / reject
        Worker->>Model: Invoke approved model
        Model-->>Worker: Model output + usage metadata
        Worker->>Worker: Validate and safety-check output
        Worker->>Store: Persist result/artifact
        Worker->>Audit: Record task outcome
        User->>API: Poll task status
        API->>Store: Read task state
        API-->>User: Status/result
    end
```

## Data handling rules

- Validate all user input before processing.
- Assign request and task identifiers.
- Store only the data needed for the task.
- Keep credentials out of payloads and logs.
- Do not send confidential/restricted data to an external provider without explicit policy approval.
- Apply retention and deletion rules to prompts, files, outputs, and derived artifacts.
- Protect object storage with scoped access and short-lived download links where applicable.
