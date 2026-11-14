from zentrax_runtime.security import InputValidator,TokenAuthenticator,DataCipher
from zentrax_runtime.policy import PolicyEngine
class SecurityCore:
    def __init__(self):
        self.validator=InputValidator(); self.auth=TokenAuthenticator()
        self.cipher=DataCipher(); self.policy=PolicyEngine()
    def validate_request(self,prompt,token):
        return {"actor":self.auth.authenticate(token),"prompt":self.validator.text(prompt,"prompt")}
