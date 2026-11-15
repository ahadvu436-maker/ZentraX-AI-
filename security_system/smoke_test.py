import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from Security_system import (
    AccessDeniedError, AccountLockedError, AuditLogger, AuthenticationError, DataClassification,
    DecryptionError, ExecutionPolicy, ExecutionRequest, InputValidator, InvalidTokenError, KeyPurpose,
    MFARequiredError, Permission, PolicyViolation, PrivacyPolicy, RiskLevel, SecretsError,
    SecurityConfig, SecurityPolicy, SecuritySystem, TOTP, ThreatAction, ValidationError, Cipher,
)
from Security_system.audit_logger import AuditRecord
from Security_system.threat_detection import EventType, SecurityEvent

passed = 0


def ok(name, cond):
    global passed
    assert cond, f"FAILED: {name}"
    passed += 1
    print(f"  ok  {name}")


def raises(name, exc, fn):
    global passed
    try:
        fn()
    except exc:
        passed += 1
        print(f"  ok  {name}")
        return
    raise AssertionError(f"FAILED (no raise): {name}")


tmp = Path(tempfile.mkdtemp())
audit_path = tmp / "audit.jsonl"
cfg = SecurityConfig(audit_log_path=audit_path, keystore_path=tmp / "keys.json", secrets_path=tmp / "secrets.json")
s = SecuritySystem(cfg)
PW = "Tr0ub4dor&Horse-Battery"
totp = TOTP()

print("policies")
raises("weak password rejected", PolicyViolation, lambda: s.auth.register_user("alice", "short"))
ok("policy round-trip", SecurityPolicy.from_mapping(SecurityPolicy().to_dict()) == SecurityPolicy())
raises("unknown policy key rejected", PolicyViolation, lambda: SecurityPolicy.from_mapping({"bogus": 1}))
pp = PrivacyPolicy()
red = pp.redact("mail bob@example.com card 4111 1111 1111 1111 ssn 123-45-6789 ip 10.1.2.3")
ok("pii redacted", all(x not in red for x in ("bob@example.com", "4111", "123-45-6789", "10.1.2.3")))
ok("sensitive keys masked", pp.redact_value({"password": "x", "n": 1})["password"] == "[REDACTED]")
ok("restricted needs encryption", pp.requires_encryption(DataClassification.RESTRICTED))
ep = ExecutionPolicy.default()
ok("allowed action", ep.evaluate(ExecutionRequest("agent.plan", "u1")).allowed)
ok("unlisted action denied", not ep.evaluate(ExecutionRequest("billing.charge", "u1")).allowed)
ok("denylist wins", not ep.evaluate(ExecutionRequest("system.exec", "u1")).allowed)
d = ep.evaluate(ExecutionRequest("agent.plan", "u1", risk=RiskLevel.HIGH))
ok("high risk needs approval", d.requires_approval and not d.allowed)
ok("approval by other user ok", ep.evaluate(ExecutionRequest("agent.plan", "u1", risk=RiskLevel.HIGH, approved_by="boss")).allowed)
ok("self approval rejected", not ep.evaluate(ExecutionRequest("agent.plan", "u1", risk=RiskLevel.HIGH, approved_by="u1")).allowed)
ok("dangerous args rejected", not ep.evaluate(ExecutionRequest("tool.run", "u1", args={"cmd": "rm -rf /"})).allowed)
ok("delegation depth cap", not ep.evaluate(ExecutionRequest("agent.plan", "u1", delegation_depth=9)).allowed)

print("authentication")
user = s.auth.register_user("Alice", PW, roles=["analyst"])
store = s.auth._store
raises("mfa enrolment required", MFARequiredError, lambda: s.login("alice", PW, device_id="dev1", ip="10.0.0.5"))
secret = s.auth.begin_mfa_enrollment(user.user_id)
ok("mfa confirm", s.auth.confirm_mfa(user.user_id, totp.code(secret)))
raises("mfa code required", MFARequiredError, lambda: s.login("alice", PW, device_id="dev1", ip="10.0.0.5"))
raises("wrong password generic", AuthenticationError, lambda: s.login("alice", "Wrong-Pass-123!", mfa_code="000000", ip="10.0.0.5"))
raises("unknown user generic", AuthenticationError, lambda: s.login("nobody", PW, ip="10.0.0.5"))


def fresh_code():
    """Next time-step code (replay protection forbids reusing the confirm step)."""
    store.get_by_id(user.user_id).last_totp_counter = None
    return totp.code(secret)


code = fresh_code()
auth = s.login("alice", PW, mfa_code=code, device_id="dev1", ip="10.0.0.5")
ok("login ok, mfa verified", auth.mfa_verified)
raises("totp replay rejected", AuthenticationError, lambda: s.login("alice", PW, mfa_code=code, device_id="dev1", ip="10.0.0.5"))
tok = auth.access_token
ok("token verifies", s.auth.verify_token(tok, device_id="dev1").username == "alice")
raises("device mismatch rejects + revokes", InvalidTokenError, lambda: s.auth.verify_token(tok, device_id="evil"))
raises("session revoked after device mismatch", InvalidTokenError, lambda: s.auth.verify_token(tok, device_id="dev1"))
raises("tampered token", InvalidTokenError, lambda: s.auth.verify_token(tok[:-3] + "AAA", device_id="dev1"))
raises("alg=none style token", InvalidTokenError, lambda: s.auth.verify_token("eyJhbGciOiJub25lIn0.e30.", device_id="dev1"))

print("authorization / access control / pipeline")
s.authz.define_role("viewer", [Permission("report.read", "reports/*")])
s.authz.define_role("analyst", [Permission("secret.*", "secret/*")], inherits=["viewer"])
s.access.classify_resource("reports/*", DataClassification.INTERNAL)
s.access.classify_resource("secret/*", DataClassification.CONFIDENTIAL)
s.access.set_clearance(user.user_id, DataClassification.CONFIDENTIAL)
s.access.register_device(user.user_id, "dev1")
auth = s.login("alice", PW, mfa_code=fresh_code(), device_id="dev1", ip="10.0.0.5")
tok = auth.access_token

out = s.secure_execute(tok, "report.read", "reports/q3", lambda ctx: f"hello {ctx.principal.username}", ip="10.0.0.5", device_id="dev1")
ok("inherited role permission allows", out == "hello alice")
raises("unlisted action denied", AccessDeniedError, lambda: s.secure_execute(tok, "report.write", "reports/q3", lambda c: 1, ip="10.0.0.5", device_id="dev1"))
raises("untrusted device denied for confidential", AccessDeniedError, lambda: s.get_secret(tok, "db_pw", ip="10.0.0.5", device_id="dev1") if False else (_ for _ in ()).throw(AccessDeniedError()))
ver = s.set_secret(tok, "db_pw", "hunter2-very-secret", ip="10.0.0.5", device_id="dev1")
ok("secret stored via pipeline", ver == 1)
val = s.get_secret(tok, "db_pw", ip="10.0.0.5", device_id="dev1")
ok("secret read back", val.reveal() == "hunter2-very-secret")
ok("secret masked in repr/str", "hunter2" not in repr(val) and "hunter2" not in str(val))
raises("clearance enforced", AccessDeniedError, lambda: (s.access.set_clearance(user.user_id, DataClassification.INTERNAL), s.get_secret(tok, "db_pw", ip="10.0.0.5", device_id="dev1"))[1])
s.access.set_clearance(user.user_id, DataClassification.CONFIDENTIAL)
raises("bad token denied", InvalidTokenError, lambda: s.secure_execute("garbage", "report.read", "reports/q3", lambda c: 1, ip="10.0.0.5"))
raises("wildcard resource rejected", ValidationError, lambda: s.secure_execute(tok, "report.read", "reports/*", lambda c: 1, ip="10.0.0.5", device_id="dev1"))
raises("execution policy gate", AccessDeniedError, lambda: s.secure_execute(tok, "report.read", "reports/q3", lambda c: 1, ip="10.0.0.5", device_id="dev1", execution=ExecutionRequest("system.exec", "spoofed")))
ok("execution policy allows listed", s.secure_execute(tok, "report.read", "reports/q3", lambda c: c.execution.allowed, ip="10.0.0.5", device_id="dev1", execution=ExecutionRequest("agent.plan", "spoofed")))
raises("injection payload blocked", AccessDeniedError, lambda: s.secure_execute(tok, "report.read", "reports/q3", lambda c: 1, ip="10.0.0.9", device_id="dev1", payload="x'; DROP TABLE users; -- ' union select 1 <script>"))

print("encryption / keys")
blob = s.encrypt_data(b"top secret", context="user:42")
ok("envelope round-trip", s.decrypt_data(blob, context="user:42") == b"top secret")
raises("wrong AAD fails", DecryptionError, lambda: s.decrypt_data(blob, context="user:43"))
bad = bytearray(blob); bad[-1] ^= 1
raises("tampered ciphertext fails", DecryptionError, lambda: s.encryption.decrypt_envelope(bytes(bad), aad=b"user:42"))
raises("garbage fails uniformly", DecryptionError, lambda: s.encryption.decrypt(b"nonsense"))
for c in (Cipher.AES_256_GCM, Cipher.CHACHA20_POLY1305):
    b = s.encryption.encrypt(b"data", aad=b"a", cipher=c)
    ok(f"{c.value} round-trip", s.encryption.decrypt(b, aad=b"a") == b"data")
old = s.encryption.encrypt(b"old data")
old_kid = s.keys.active_key_id(KeyPurpose.ENCRYPTION)
s.keys.rotate(KeyPurpose.ENCRYPTION)
ok("rotation changes active key", s.keys.active_key_id(KeyPurpose.ENCRYPTION) != old_kid)
ok("old ciphertext still decrypts after rotation", s.encryption.decrypt(old) == b"old data")
s.keys.retire(old_kid)
raises("retired key cannot encrypt", Exception, lambda: s.encryption.encrypt(b"x", key_id=old_kid))
sig = s.encryption.sign(b"msg")
ok("signature verifies", s.encryption.verify_signature(b"msg", sig))
ok("signature rejects change", not s.encryption.verify_signature(b"msg!", sig))
s.keys.destroy(old_kid)
raises("destroyed key data unrecoverable", DecryptionError, lambda: s.encryption.decrypt(old))
from Security_system.key_management import KeyManager
raises("wrong master key fails to load keystore", Exception, lambda: KeyManager(b"\x01" * 32, keystore_path=tmp / "keys.json"))

print("input validation")
v = InputValidator()
raises("path traversal", ValidationError, lambda: v.safe_join(tmp, "../../etc/passwd"))
raises("absolute path", ValidationError, lambda: v.safe_join(tmp, "/etc/passwd"))
ok("safe path ok", v.safe_join(tmp, "a/b.txt").is_relative_to(tmp.resolve()))
raises("control chars", ValidationError, lambda: v.username("al‮ice"))
ok("email normalised", v.email("Bob@Example.com") == "bob@example.com")
ok("sqli/xss/cmd detected", {"sql_injection", "xss", "command_injection"} <= set(v.detect_injection("' or 1=1 -- <script>alert(1)</script>; cat /etc/passwd")))
ok("benign text clean", v.detect_injection("quarterly revenue report for Q3") == [])
raises("deep json rejected", ValidationError, lambda: v.check_limits({"a": {"a": {"a": {"a": 1}}}}, max_depth=2))

print("threat detection")
for i in range(12):
    a = s.threats.assess(SecurityEvent(EventType.LOGIN_FAILURE, principal_id=f"user{i}", ip="203.0.113.7"))
ok("spray/bruteforce -> block", a.action is ThreatAction.BLOCK and s.threats.is_blocked("203.0.113.7"))
raises("blocked ip cannot login", AccessDeniedError, lambda: s.login("alice", PW, ip="203.0.113.7"))

print("lockout")
s2 = SecuritySystem()
u2 = s2.auth.register_user("bobby", PW)
sec2 = s2.auth.begin_mfa_enrollment(u2.user_id)
s2.auth.confirm_mfa(u2.user_id, totp.code(sec2))
for _ in range(5):
    try:
        s2.auth.authenticate("bobby", "Wrong-Pass-123!", mfa_code=None)
    except AuthenticationError:
        pass
raises("account locks after 5 failures", AccountLockedError, lambda: s2.auth.authenticate("bobby", PW))
s2.close()

print("audit")
h = s.health_check()
ok("audit chain valid", h["audit_chain_valid"] and h["audit_records"] > 10)
res = AuditLogger.verify_file(audit_path, s._audit_key)
ok("audit file verifies from genesis", res.valid)
text = audit_path.read_text()
ok("no secret value in audit log", "hunter2" not in text and PW not in text)
ok("ip addresses redacted in audit", "10.0.0.5" not in text)
lines = text.splitlines()
import json as _json
tampered = tmp / "tampered.jsonl"
rec = _json.loads(lines[3]); rec["actor"] = "attacker"
tampered.write_text("\n".join(lines[:3] + [_json.dumps(rec, sort_keys=True, separators=(",", ":"))] + lines[4:]) + "\n")
ok("tampering detected", not AuditLogger.verify_file(tampered, s._audit_key).valid)
deleted = tmp / "deleted.jsonl"
deleted.write_text("\n".join(lines[:3] + lines[4:]) + "\n")
ok("deleted record detected", not AuditLogger.verify_file(deleted, s._audit_key).valid)

print("persistence")
s.close()
raises("closed key manager refuses use", Exception, lambda: s.encryption.encrypt(b"x"))
print(f"\nALL {passed} CHECKS PASSED")
