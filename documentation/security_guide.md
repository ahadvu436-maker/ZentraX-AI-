# ZentraX AI — Security Guide

**Status:** Security baseline. This guide is not a certification or penetration-test report.

## 1. Security objectives

Protect confidentiality, integrity, availability, accountability, and user control. Security claims must be supported by tests, configuration evidence, and independent review.

Never claim that hackers cannot steal any data or that the system is unbreakable. State the controls implemented, their test coverage, and residual risks.

## 2. Threat model

Assets:
- source code and intellectual property;
- user prompts, files, and outputs;
- credentials, tokens, keys, and certificates;
- model weights and configuration;
- task and audit records;
- infrastructure and deployment pipeline.

Threat actors:
- unauthenticated internet attacker;
- compromised user account;
- malicious or careless insider;
- compromised dependency or build pipeline;
- malicious uploaded file or prompt;
- compromised model/provider;
- misconfigured administrator;
- compromised worker or agent.

Threat scenarios:
- credential theft;
- broken access control / cross-tenant access;
- prompt injection and tool misuse;
- unsafe file parsing;
- SSRF and arbitrary URL access;
- command injection;
- dependency/supply-chain compromise;
- secrets leaked in logs or repository;
- denial of service and resource exhaustion;
- model data leakage;
- unauthorized autonomous action;
- backup theft or ransomware.

## 3. Identity and authentication

- Use a maintained authentication framework.
- Hash passwords with a modern password-hashing algorithm and safe parameters.
- Support MFA for privileged users where feasible.
- Rate-limit login and recovery operations.
- Use secure session/token lifetimes.
- Rotate refresh tokens where supported.
- Revoke sessions on password reset or suspected compromise.
- Do not store plaintext passwords.
- Do not invent custom cryptography.

## 4. Authorization

Use deny-by-default authorization.
- Check permissions on every protected operation.
- Enforce tenant and object ownership boundaries.
- Separate user, operator, security administrator, and system roles.
- Use scoped service identities for workers.
- Review privilege escalation paths.
- Record privileged changes in an append-only or tamper-evident audit system where feasible.

A hidden UI button is not an authorization control. The backend must enforce the permission.

## 5. Zero Trust implementation principles

Zero Trust is a continuous policy model, not a single product or folder.

For each request:
1. identify the principal and workload;
2. authenticate identity;
3. validate device/workload posture where supported;
4. evaluate action, resource, context, and policy;
5. grant only narrow, short-lived access;
6. log the decision;
7. re-evaluate when context or risk changes.

Do not trust network location alone.

## 6. Encryption and key management

- Use TLS for network communication.
- Use established libraries and algorithms.
- Encrypt sensitive stored data where required by the threat model.
- Keep keys separate from encrypted data.
- Use a KMS or secrets manager where possible.
- Rotate keys with a documented plan.
- Restrict key access and audit it.
- Back up recovery material securely.
- Do not write homemade cryptographic algorithms.
- Do not hardcode encryption keys in source code or `__init__.py`.

## 7. Input and file security

- Validate type, size, encoding, and structure.
- Use allowlists where possible.
- Treat filenames and paths as untrusted.
- Store uploads outside executable/static code paths.
- Generate server-side object identifiers.
- Scan files where appropriate.
- Parse complex files in isolated workers.
- Apply decompression limits to prevent archive bombs.
- Avoid shell execution on user-provided content.
- Prevent SSRF by restricting outbound destinations and resolving/validating addresses safely.

## 8. AI-specific security

### Prompt injection
- Treat retrieved documents and user content as untrusted data.
- Separate system policy from retrieved content.
- Do not allow retrieved text to override policy.
- Restrict tools by explicit allowlist.
- Require approval for consequential actions.

### Tool use
- Validate tool arguments.
- Enforce per-tool permissions.
- Apply timeouts and resource limits.
- Log tool invocation metadata, not secrets.
- Sandbox code execution.
- Block arbitrary network and filesystem access unless specifically authorized.

### Model output
- Treat model output as untrusted.
- Validate structured output against schemas.
- Escape content before rendering.
- Never execute generated code automatically in production.
- Apply policy checks before external side effects.

### Data privacy
- Do not send restricted data to external providers without explicit authorization and policy review.
- Record provider, region, retention policy, and data handling terms.
- Minimize prompts and redact identifiers where practical.
- Define retention and deletion controls.

## 9. Secrets management

- Use environment injection or a secrets manager.
- Commit only `.env.example` with dummy placeholders.
- Scan repository history for accidentally committed credentials.
- Rotate any real credential that may have been exposed.
- Avoid secrets in command-line arguments and logs.
- Use separate development/staging/production credentials.

## 10. Logging and audit

Log:
- actor/service identity;
- action;
- target resource identifier;
- allow/deny outcome;
- timestamp;
- correlation ID;
- policy version;
- relevant security context.

Do not log:
- passwords;
- access/refresh tokens;
- private keys;
- full payment credentials;
- raw sensitive prompts/files by default.

Protect audit logs from unauthorized alteration and define retention.

## 11. Dependency and supply-chain security

- Pin dependencies.
- Review dependency licenses.
- Scan known vulnerabilities.
- Use trusted package sources.
- Protect repository branches and CI secrets.
- Require review for production changes.
- Generate SBOM where feasible.
- Sign release artifacts where feasible.
- Rebuild from clean, reproducible inputs.

## 12. Security testing plan

- static application security testing;
- dependency vulnerability scanning;
- secret scanning;
- container image scanning;
- unit tests for authentication/authorization;
- integration tests for tenant isolation;
- API fuzzing and schema validation;
- file upload and parser tests;
- rate-limit and denial-of-service resilience tests;
- prompt-injection and tool-abuse tests;
- manual threat-model review;
- independent penetration test before high-risk production launch.

## 13. Incident response

Maintain a runbook for:
- compromised user account;
- leaked API key;
- suspected source repository compromise;
- malicious uploaded file;
- data exposure;
- ransomware;
- compromised model/provider;
- unauthorized agent action;
- denial of service.

Each runbook must identify decision owner, containment steps, evidence to preserve, recovery process, and notification obligations.

## 14. Security acceptance criteria

A security feature is not complete because a function exists. It must have:
- documented threat addressed;
- secure configuration;
- negative tests;
- positive tests;
- logging/audit behavior;
- failure behavior;
- review evidence;
- known limitations.

## 15. Compliance note

Applicable privacy, consumer, cybersecurity, export, and AI laws depend on product function, customer location, data type, and deployment model. Obtain qualified legal advice before making compliance claims or entering regulated markets.
