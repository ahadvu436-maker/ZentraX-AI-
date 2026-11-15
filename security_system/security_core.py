"""Central orchestrator binding every Security_system module together.

Request pipeline (``SecuritySystem.secure_execute``)::

    1. Threat screen (blocklist, rate limit, payload signatures)   -> threat_detection
    2. Input validation of action / resource names                  -> input_validation
    3. Token + live-session verification, device binding            -> authentication
    4. Contextual access control (classification, clearance,
       device, network, MFA, risk) + RBAC/ABAC/ACL                  -> access_control
    5. Execution policy (agents / tools), actor forced to caller    -> policies.execution_policy
    6. Audit the decision *before* acting (fail closed)             -> audit_logger
    7. Run the operation, then audit the outcome                    -> audit_logger

Anything unexpected is treated as a denial. Key material for tokens and the
audit chain is derived from the master key with HKDF using distinct labels, so
compromise of one subsystem key does not expose the others.
"""

from __future__ import annotations

import logging
import secrets
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Callable, TypeVar

from .access_control import AccessContext, AccessController, AccessDecision, AccessDeniedError, AccessRequest
from .audit_logger import AuditError, AuditLogger, AuditOutcome, AuditSeverity, FileAuditSink
from .authentication import (
    AuthenticationError,
    AuthenticationService,
    AuthResult,
    InMemoryUserStore,
    InvalidTokenError,
    MFARequiredError,
    Principal,
    TokenService,
    UserStore,
)
from .authorization import AuthorizationEngine
from .encryption import EncryptionService
from .input_validation import InputValidator, ValidationError
from .key_management import KeyManager, load_master_key_from_env
from .policies import (
    DataClassification,
    ExecutionDecision,
    ExecutionPolicy,
    ExecutionRequest,
    PrivacyPolicy,
    SecurityError,
    SecurityPolicy,
)
from .secrets_manager import FileSecretBackend, InMemorySecretBackend, SecretsManager, SecretStr
from .threat_detection import EventType, SecurityEvent, ThreatAction, ThreatAssessment, ThreatDetector

__all__ = ["SecurityConfig", "SecurityContext", "SecuritySystem"]

T = TypeVar("T")
_log = logging.getLogger("Security_system")


@dataclass(frozen=True)
class SecurityConfig:
    """Everything needed to assemble a :class:`SecuritySystem`.

    If ``master_key`` is omitted an *ephemeral* random key is generated. That is
    fine for tests and demos but means keys, tokens and the audit chain cannot
    survive a restart. In production supply the key from an HSM/KMS or via
    :meth:`SecuritySystem.from_environment`.
    """

    policy: SecurityPolicy = field(default_factory=SecurityPolicy)
    privacy: PrivacyPolicy = field(default_factory=PrivacyPolicy)
    execution: ExecutionPolicy = field(default_factory=ExecutionPolicy.default)
    master_key: bytes | None = field(default=None, repr=False)
    keystore_path: Path | None = None
    audit_log_path: Path | None = None
    secrets_path: Path | None = None
    user_store: UserStore | None = None


@dataclass(frozen=True)
class SecurityContext:
    """Verified facts about the caller, handed to the protected operation."""

    principal: Principal
    ip: str | None
    device_id: str | None
    decision: AccessDecision
    execution: ExecutionDecision | None = None


class SecuritySystem:
    """Facade and policy enforcement point for the whole package."""

    def __init__(self, config: SecurityConfig | None = None) -> None:
        self.config = config or SecurityConfig()
        policy = self.config.policy

        master = self.config.master_key
        self.ephemeral = master is None
        if master is None:
            master = secrets.token_bytes(32)
            _log.warning("SecuritySystem started with an EPHEMERAL master key; do not use in production")
        if len(master) != 32:
            raise SecurityError("master key must be exactly 32 bytes")

        token_key = EncryptionService.hkdf_derive(master, info=b"security_system/token-signing/v1")
        audit_key = EncryptionService.hkdf_derive(master, info=b"security_system/audit-mac/v1")

        self.validator = InputValidator()
        self.keys = KeyManager(
            master,
            keystore_path=self.config.keystore_path,
            rotation_interval_days=policy.crypto.key_rotation_days,
        )
        self._audit_key = audit_key
        self._audit_sink = FileAuditSink(self.config.audit_log_path) if self.config.audit_log_path else None
        self.audit = AuditLogger(audit_key, sink=self._audit_sink, privacy=self.config.privacy)
        self.encryption = EncryptionService(self.keys, policy.crypto)
        backend = FileSecretBackend(self.config.secrets_path) if self.config.secrets_path else InMemorySecretBackend()
        self.secrets = SecretsManager(self.encryption, backend, audit=self.audit)
        self.authz = AuthorizationEngine()
        self.access = AccessController(policy, self.authz)
        self.threats = ThreatDetector(policy, validator=self.validator)
        self.auth = AuthenticationService(
            policy,
            self.config.user_store or InMemoryUserStore(),
            TokenService(
                token_key,
                ttl_seconds=policy.session.access_token_ttl_seconds,
                clock_skew_seconds=policy.session.max_clock_skew_seconds,
            ),
        )
        self.audit.log("system.start", actor="system", action="start", details={"ephemeral_key": self.ephemeral})

    # --------------------------------------------------------------- factories
    @classmethod
    def from_environment(
        cls,
        *,
        env_var: str = "SECURITY_MASTER_KEY",
        base_config: SecurityConfig | None = None,
    ) -> SecuritySystem:
        """Build a system whose master key comes from ``env_var`` (base64, 32 bytes)."""
        base = base_config or SecurityConfig()
        return cls(replace(base, master_key=load_master_key_from_env(env_var)))

    def __enter__(self) -> SecuritySystem:
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()

    def close(self) -> None:
        """Wipe key material (best effort) and close sinks."""
        try:
            self.audit.log("system.stop", actor="system", action="stop")
        except AuditError:
            _log.exception("could not write shutdown audit record")
        self.keys.close()
        if self._audit_sink is not None:
            self._audit_sink.close()

    # ----------------------------------------------------------------- login
    def login(
        self,
        username: str,
        password: str,
        *,
        mfa_code: str | None = None,
        device_id: str | None = None,
        ip: str | None = None,
    ) -> AuthResult:
        """Authenticate with threat screening and full auditing."""
        safe_ip = self._clean_ip(ip)
        actor = self.validator.sanitize_for_log(username, max_length=64) if isinstance(username, str) else "-"
        pre = self.threats.assess(SecurityEvent(EventType.REQUEST, ip=safe_ip))
        if pre.action is ThreatAction.BLOCK:
            self._audit_best_effort(
                "auth.login", actor, "login", "-", AuditOutcome.DENIED, AuditSeverity.WARNING,
                {"ip": safe_ip, "indicators": list(pre.indicators)},
            )
            raise AccessDeniedError("Access denied")
        try:
            result = self.auth.authenticate(username, password, mfa_code=mfa_code, device_id=device_id, ip=safe_ip)
        except MFARequiredError:
            self._audit_best_effort("auth.login", actor, "login", "-", AuditOutcome.FAILURE, AuditSeverity.INFO,
                                    {"ip": safe_ip, "reason": "mfa_required"})
            raise
        except AuthenticationError as exc:
            principal_key = actor.lower()
            assessment = self.threats.assess(SecurityEvent(EventType.LOGIN_FAILURE, principal_id=principal_key, ip=safe_ip))
            self._audit_best_effort(
                "auth.login", actor, "login", "-", AuditOutcome.FAILURE, AuditSeverity.WARNING,
                {"ip": safe_ip, "reason": type(exc).__name__, "threat": assessment.level.name},
            )
            raise
        self.threats.assess(SecurityEvent(EventType.LOGIN_SUCCESS, principal_id=result.user_id, ip=safe_ip))
        self.audit.log("auth.login", actor=result.user_id, action="login",
                       details={"ip": safe_ip, "session": result.session_id, "mfa": result.mfa_verified})
        return result

    def logout(self, token: str) -> None:
        self.auth.logout(token)

    # ------------------------------------------------------- guarded execution
    def check_access(
        self,
        token: str,
        action: str,
        resource: str,
        *,
        ip: str | None = None,
        device_id: str | None = None,
        payload: str | None = None,
        execution: ExecutionRequest | None = None,
    ) -> SecurityContext:
        """Run steps 1-6 of the pipeline. Raises on denial; returns the verified context."""
        safe_ip = self._clean_ip(ip)
        actor = "anonymous"
        try:
            # 1. threat screen (before spending any effort on the request)
            pre = self.threats.assess(SecurityEvent(EventType.REQUEST, ip=safe_ip, payload=payload))
            if pre.action is ThreatAction.BLOCK:
                self._deny(actor, action, resource, ["threat screen"], {"indicators": list(pre.indicators)})

            # 2. validate names
            action = self.validator.action_name(action)
            resource = self.validator.resource_name(resource)

            # 3. verify identity (token + live session) on *every* call
            try:
                principal = self.auth.verify_token(token, device_id=device_id, ip=safe_ip)
            except InvalidTokenError:
                self.threats.assess(SecurityEvent(EventType.TOKEN_INVALID, ip=safe_ip))
                self._audit_best_effort("access.decision", actor, action, resource, AuditOutcome.DENIED,
                                        AuditSeverity.WARNING, {"ip": safe_ip, "reason": "invalid_token"})
                raise
            actor = principal.user_id

            # 4. contextual access control
            decision = self.access.evaluate(
                AccessRequest(
                    principal_id=principal.user_id,
                    roles=principal.roles,
                    action=action,
                    resource=resource,
                    context=AccessContext(
                        ip=safe_ip,
                        device_id=device_id,
                        mfa_verified=principal.mfa_verified,
                        risk_score=pre.score,
                        attributes=principal.attributes,
                    ),
                )
            )
            if not decision.allowed:
                self.threats.assess(SecurityEvent(EventType.AUTHZ_DENIED, principal_id=actor, ip=safe_ip))
                self._deny(actor, action, resource, list(decision.reasons), {"obligations": list(decision.obligations)})

            # 5. execution policy: the actor is always the verified caller, never caller-supplied
            exec_decision: ExecutionDecision | None = None
            if execution is not None:
                exec_decision = self.config.execution.evaluate(replace(execution, actor=principal.user_id))
                if not exec_decision.allowed:
                    self._deny(actor, action, resource, list(exec_decision.reasons), {"stage": "execution_policy"})

            # 6. audit the decision *before* acting; failure to audit means no action
            self.audit.log(
                "access.decision",
                actor=actor,
                action=action,
                resource=resource,
                details={"classification": decision.classification.name, "risk": pre.score, "ip": safe_ip},
            )
            return SecurityContext(principal, safe_ip, device_id, decision, exec_decision)

        except (SecurityError, ValidationError):
            raise
        except Exception as exc:  # noqa: BLE001 - fail closed on any unexpected error
            _log.exception("unexpected error in access pipeline")
            self._audit_best_effort("access.decision", actor, action if isinstance(action, str) else "-",
                                    resource if isinstance(resource, str) else "-", AuditOutcome.ERROR,
                                    AuditSeverity.ERROR, {"error": type(exc).__name__})
            raise AccessDeniedError("Access denied") from exc

    def secure_execute(
        self,
        token: str,
        action: str,
        resource: str,
        operation: Callable[[SecurityContext], T],
        *,
        ip: str | None = None,
        device_id: str | None = None,
        payload: str | None = None,
        execution: ExecutionRequest | None = None,
    ) -> T:
        """Verify, authorise, audit, then run ``operation(context)`` (step 7 audits the outcome)."""
        context = self.check_access(
            token, action, resource, ip=ip, device_id=device_id, payload=payload, execution=execution
        )
        actor = context.principal.user_id
        try:
            result = operation(context)
        except Exception as exc:
            self._audit_best_effort("access.completed", actor, action, resource, AuditOutcome.ERROR,
                                    AuditSeverity.ERROR, {"error": type(exc).__name__})
            raise
        self._audit_best_effort("access.completed", actor, action, resource, AuditOutcome.SUCCESS,
                                AuditSeverity.INFO, {})
        return result

    # ------------------------------------------------------ guarded conveniences
    def get_secret(self, token: str, name: str, *, ip: str | None = None, device_id: str | None = None) -> SecretStr:
        """Read a secret through the full pipeline (resource ``secret/<name>``)."""
        return self.secure_execute(
            token, "secret.read", f"secret/{name}",
            lambda ctx: self.secrets.get_secret(name, actor=ctx.principal.user_id),
            ip=ip, device_id=device_id,
        )

    def set_secret(self, token: str, name: str, value: str, *, ip: str | None = None,
                   device_id: str | None = None, ttl_seconds: float | None = None) -> int:
        """Write a secret through the full pipeline; returns the new version."""
        meta = self.secure_execute(
            token, "secret.write", f"secret/{name}",
            lambda ctx: self.secrets.set_secret(name, value, actor=ctx.principal.user_id, ttl_seconds=ttl_seconds),
            ip=ip, device_id=device_id,
        )
        return meta.version

    # ---------------------------------------------------------- data protection
    def encrypt_data(self, data: bytes, *, context: str, classification: DataClassification = DataClassification.CONFIDENTIAL) -> bytes:
        """Envelope-encrypt ``data``; ``context`` (e.g. ``"user:42:profile"``) is bound as AAD."""
        if not self.config.privacy.requires_encryption(classification):
            _log.info("encrypting %s data although policy does not require it", classification.name)
        return self.encryption.encrypt_envelope(data, aad=context.encode("utf-8"))

    def decrypt_data(self, blob: bytes, *, context: str) -> bytes:
        return self.encryption.decrypt_envelope(blob, aad=context.encode("utf-8"))

    def redact(self, value: Any) -> Any:
        """Apply the privacy policy's redaction to text or structures."""
        return self.config.privacy.redact_value(value)

    # --------------------------------------------------------------------- ops
    def health_check(self) -> dict[str, Any]:
        """Operational snapshot for monitoring (contains no secrets)."""
        chain = self.audit.verify()
        seq, _ = self.audit.head()
        return {
            "ephemeral_master_key": self.ephemeral,
            "audit_chain_valid": chain.valid,
            "audit_records": seq,
            "keys": self.keys.list_keys(),
            "keys_due_for_rotation": [k["key_id"] for k in self.keys.list_keys() if self.keys.needs_rotation(k["key_id"])],
        }

    # ---------------------------------------------------------------- internals
    def _clean_ip(self, ip: str | None) -> str | None:
        if ip is None:
            return None
        try:
            return self.validator.ip_address(ip)
        except ValidationError:
            return None  # an unparseable source is treated as unknown, which zero-trust handles

    def _deny(self, actor: str, action: str, resource: str, reasons: list[str], extra: dict[str, Any]) -> None:
        self._audit_best_effort("access.decision", actor, str(action), str(resource), AuditOutcome.DENIED,
                                AuditSeverity.WARNING, {"reasons": reasons, **extra})
        raise AccessDeniedError("Access denied")

    def _audit_best_effort(self, event: str, actor: str, action: str, resource: str,
                           outcome: AuditOutcome, severity: AuditSeverity, details: dict[str, Any]) -> None:
        """For denial/post-action records, where a logging failure must not mask the real outcome."""
        try:
            self.audit.log(event, actor=actor, action=action, resource=resource,
                           outcome=outcome, severity=severity, details=details)
        except AuditError:
            _log.exception("audit write failed for event %s", event)
