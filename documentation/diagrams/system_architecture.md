# System Architecture Diagram

This is the intended logical architecture. It is not evidence that all boxes are implemented.

```mermaid
flowchart TB
    U[User / Client] -->|HTTPS| IN[Ingress / Reverse Proxy]
    IN --> API[Backend API]
    API --> AUTH[Authentication]
    API --> POL[Authorization and Policy]
    API --> VAL[Schema and Input Validation]
    AUTH --> POL
    POL --> APP[Application Services]
    VAL --> APP

    APP --> ORCH[Task Orchestrator]
    ORCH --> Q[Queue / Task Broker]
    Q --> W[Worker Pool]
    APP --> REG[Model Registry]
    W --> REG

    REG --> ADAPT[Model Provider Adapters]
    ADAPT --> LOCAL[Local CPU/GPU Runtime]
    ADAPT --> EXT[Approved External Provider]

    APP --> DB[(Relational Database)]
    ORCH --> DB
    W --> DB
    APP --> CACHE[(Cache)]
    Q --> CACHE
    W --> OBJ[(Object / Artifact Storage)]

    APP --> AUD[Audit Events]
    AUTH --> AUD
    POL --> AUD
    W --> AUD

    API --> TEL[Logs / Metrics / Traces]
    W --> TEL
    LOCAL --> TEL
    AUD --> TEL

    GOV[Governance and Human Approval] --> POL
    GOV --> ORCH
    STOP[Emergency Stop] --> ORCH
    STOP --> W
```

## Trust boundaries

- Client to ingress: untrusted network.
- Ingress to API: authenticated service boundary.
- API to data stores: private service boundary.
- Worker to model runtime: restricted execution boundary.
- Platform to external provider: third-party data-processing boundary.
- Agent to tools: permissioned and sandboxed boundary.

## Design notes

- The policy decision must occur before protected actions.
- External model calls must be explicit and policy-controlled.
- Worker execution must have bounded permissions and resources.
- Audit events must not contain raw secrets.
- Emergency stop must be enforced by the execution path, not merely shown in the UI.
