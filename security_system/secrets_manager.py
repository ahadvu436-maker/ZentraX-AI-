import os
class EnvironmentSecrets:
    """Reads secrets from environment only; never logs secret values."""
    def get(self, name, required=True):
        value = os.environ.get(name)
        if required and not value: raise RuntimeError(f"required secret is not configured: {name}")
        return value
