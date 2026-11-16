# ZentraX AI Documentation

This directory contains the current architecture and engineering documentation baseline.

## Documents

- [Architecture](architecture.md) — system boundaries, principles, module responsibilities.
- [System Design](system_design.md) — runtime components, task lifecycle, integration patterns.
- [API Reference](api_reference.md) — proposed API contracts; verify against actual routes.
- [Installation](installation.md) — local, CPU, and GPU setup guidance.
- [Deployment Guide](deployment_guide.md) — container, infrastructure, rollout, and recovery.
- [Security Guide](security_guide.md) — threat model and security engineering baseline.
- [Developer Guide](developer_guide.md) — package, coding, testing, and contribution rules.
- [Roadmap](roadmap.md) — staged implementation and acceptance criteria.
- [Diagrams](diagrams/) — system architecture, data flow, and agent workflow.

## Status policy

Every feature must be labelled using evidence:
- `Implemented`: code exists and required tests pass.
- `Partial`: some functionality exists, but acceptance criteria are incomplete.
- `Planned`: designed but not implemented.
- `Not verified`: status cannot yet be established.

Do not label a feature complete based only on a folder, filename, class stub, or diagram.

## Updating this documentation

When implementation changes:
1. update the relevant document;
2. link to the implementation or interface;
3. update the test evidence;
4. record architectural decisions;
5. note known limitations.

This documentation is a target engineering baseline and must be reconciled with the live source repository before being presented as an as-built specification.
