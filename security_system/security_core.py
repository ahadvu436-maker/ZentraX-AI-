from dataclasses import dataclass
from .authentication import Authenticator
from .authorization import Authorizer
from .access_control import AccessController
from .threat_detection import ThreatDetector
from .audit_logger import AuditLogger
from .input_validation import validate_payload

@dataclass(frozen=True)
class SecurityConfig:
    max_payload_bytes: int = 1_000_000
    require_authenticated_principal: bool = True

class SecurityCore:
    def __init__(self, authenticator, authorizer, access, detector, audit, config=None):
        self.authenticator, self.authorizer, self.access = authenticator, authorizer, access
        self.detector, self.audit = detector, audit
        self.config = config or SecurityConfig()

    def inspect_request(self, token, action, resource, payload):
        validate_payload(payload, self.config.max_payload_bytes)
        principal = self.authenticator.verify(token)
        if self.config.require_authenticated_principal and principal is None:
            self.audit.write("auth.denied", {"action": action, "reason": "invalid_token"})
            raise PermissionError("authentication required")
        findings = self.detector.inspect(payload)
        if findings:
            self.audit.write("threat.detected", {"principal": principal.subject, "findings": findings})
            raise PermissionError("request rejected by threat policy")
        self.authorizer.require(principal, action)
        self.access.require(principal, resource, action)
        self.audit.write("request.allowed", {"principal": principal.subject, "action": action, "resource": resource})
        return principal
