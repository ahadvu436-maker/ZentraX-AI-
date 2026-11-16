# Agent Workflow Diagram

This is a bounded, policy-controlled agent workflow. It does not authorize unrestricted autonomous operation.

```mermaid
flowchart TD
    START[User Task] --> AUTH[Authenticate and authorize]
    AUTH --> PLAN[Create bounded task plan]
    PLAN --> CHECK{Policy allows plan?}
    CHECK -->|No| DENY[Reject and audit]
    CHECK -->|Approval required| HUMAN[Human review]
    HUMAN -->|Reject| DENY
    HUMAN -->|Approve| EXEC
    CHECK -->|Allow| EXEC[Execute one approved step]
    EXEC --> TOOL{Tool required?}
    TOOL -->|No| EVAL[Evaluate result]
    TOOL -->|Yes| TCHK[Validate tool and arguments]
    TCHK --> PERM{Permission and budget valid?}
    PERM -->|No| BLOCK[Block tool call and audit]
    PERM -->|Yes| RUN[Run in restricted environment]
    RUN --> EVAL
    BLOCK --> EVAL
    EVAL --> DONE{Goal met or limit reached?}
    DONE -->|No| LIMIT{Step/time/token budget left?}
    LIMIT -->|Yes| PLAN
    LIMIT -->|No| STOP[Stop safely and report partial result]
    DONE -->|Yes| OUTPUT[Validate output and safety]
    OUTPUT --> SAVE[Persist approved result and audit]
    SAVE --> END[Return result]
    DENY --> END
    STOP --> END
```

## Agent invariants

1. The user authorization scope is inherited; an agent cannot expand it.
2. Tool access is allowlisted.
3. Every tool call is validated and logged.
4. Step, time, token, memory, and compute budgets are enforced.
5. External side effects require approval according to risk policy.
6. Retrieved content and model output are untrusted.
7. Emergency stop blocks new execution and signals active workers.
8. Agent state and outputs are tenant-isolated.
9. A failed policy check cannot be overridden by the model.
10. Completion claims must match actual tool outcomes.
