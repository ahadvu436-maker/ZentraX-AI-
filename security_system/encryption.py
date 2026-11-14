class EncryptionProvider:
    """Delegates to vetted crypto/KMS; intentionally no custom cryptography."""
    def __init__(self, provider=None): self.provider=provider
    def encrypt(self, data, key_id):
        if self.provider is None: raise RuntimeError("Configure approved KMS/crypto provider")
        return self.provider.encrypt(data,key_id)
    def decrypt(self, data, key_id):
        if self.provider is None: raise RuntimeError("Configure approved KMS/crypto provider")
        return self.provider.decrypt(data,key_id)
