from dataclasses import dataclass

@dataclass(frozen=True)
class Permission:
    action: str
    roles: frozenset[str]

class Authorizer:
    def __init__(self, permissions=None):
        self.permissions = {p.action: p.roles for p in (permissions or [])}
    def allow(self, action, roles):
        self.permissions[action] = frozenset(roles)
    def require(self, principal, action):
        if principal is None: raise PermissionError("principal missing")
        allowed = self.permissions.get(action, frozenset())
        if not allowed.intersection(principal.roles):
            raise PermissionError(f"role not permitted for action: {action}")
        return True
