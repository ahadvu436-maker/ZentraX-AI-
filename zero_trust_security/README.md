# ZentraX Zero-Trust Security — development baseline

This package creates the 41 requested top-level module directories and provides
consistent Python service boundaries plus a small default-deny policy core.

## Important status
This is a **development scaffold / baseline**, not a production-ready security product,
certification, or guarantee against data theft. Several modules are adapter boundaries
and require real provider integrations, threat models, implementation, independent review,
and deployment-specific enforcement before use.

## Install and test
Python 3.11+
```bash
python -m venv .venv
# Windows: .venv\\Scripts\\activate
# macOS/Linux: source .venv/bin/activate
pip install -e ".[dev]"
pytest
```

## Example
```python
from zentrax_zero_trust_security.core import PolicyEngine, SecurityContext

engine = PolicyEngine()
result = engine.evaluate(SecurityContext(
    subject="agent-01",
    action="read",
    resource="customer-profile",
    roles=("user",),
    risk_score=0.1,
))
print(result.decision, result.reason)
```

## Before production
- Integrate a vetted OIDC identity provider and WebAuthn/TOTP MFA.
- Use a managed secrets vault and KMS/HSM; do not invent cryptography.
- Persist policies with versioning, approvals, and rollback.
- Protect audit logs against tampering and restrict access.
- Add threat modelling, fuzzing, dependency scanning, SAST/DAST, penetration testing.
- Enforce network segmentation and workload identity in the actual infrastructure.
- Keep high-impact containment, key rotation, and privilege changes behind human approval.
- Run only authorized security tests in isolated local/staging environments.
