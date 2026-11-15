class AccessController:
    def __init__(self, rules=None):
        self.rules = list(rules or [])
    def add_rule(self, subject, resource_prefix, actions):
        self.rules.append((subject, resource_prefix, frozenset(actions)))
    def require(self, principal, resource, action):
        for subject, prefix, actions in self.rules:
            if subject == principal.subject and resource.startswith(prefix) and action in actions:
                return True
        raise PermissionError("resource access denied")
