from .security_core import SecurityCore
class AuthorizationService:
    def __init__(self, core=None): self.core=core or SecurityCore()
    def authorize(self, request): return self.core.evaluate(request)
