def require_text(value, field="value", max_length=4096):
    if not isinstance(value,str) or not value.strip(): raise ValueError(f"{field} must be non-empty text")
    if len(value)>max_length: raise ValueError(f"{field} too long")
    return value
