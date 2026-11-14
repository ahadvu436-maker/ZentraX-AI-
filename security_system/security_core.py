from dataclasses import dataclass
from enum import Enum
from datetime import datetime, timezone
import json, os

class Decision(str, Enum):
    ALLOW="allow"; DENY="deny"; APPROVAL="require_approval"

@dataclass(frozen=True)
class SecurityRequest:
    subject: str
    action: str
    resource: str
    roles: tuple[str,...]=()
    risk: float=1.0
    mfa: bool=False

@dataclass(frozen=True)
class SecurityResult:
    decision: Decision
    reason: str
    timestamp: str

class SecurityCore:
    SENSITIVE={"delete","admin","export_all","deploy","rotate_root_key"}
    def __init__(self, audit_path="var/security_audit.jsonl"):
        self.audit_path=audit_path
    def evaluate(self, req: SecurityRequest):
        if not req.subject or not req.action or not req.resource:
            d,r=Decision.DENY,"incomplete request"
        elif not 0 <= req.risk <= 1:
            d,r=Decision.DENY,"invalid risk score"
        elif req.risk >= .8:
            d,r=Decision.DENY,"high risk"
        elif req.action in self.SENSITIVE or req.risk >= .5:
            d,r=Decision.APPROVAL,"human approval required"
        elif req.action in {"read","write","execute","invoke"} and "user" in req.roles and req.mfa:
            d,r=Decision.ALLOW,"baseline policy passed"
        else:
            d,r=Decision.DENY,"no explicit allow policy"
        ts=datetime.now(timezone.utc).isoformat()
        os.makedirs(os.path.dirname(self.audit_path) or ".",exist_ok=True)
        with open(self.audit_path,"a",encoding="utf-8") as f:
            f.write(json.dumps({"subject":req.subject,"action":req.action,"resource":req.resource,"decision":d.value,"reason":r,"timestamp":ts})+"\\n")
        return SecurityResult(d,r,ts)
