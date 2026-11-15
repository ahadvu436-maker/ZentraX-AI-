import json
def validate_payload(payload, max_bytes=1_000_000):
    try: encoded = json.dumps(payload, ensure_ascii=False, allow_nan=False).encode("utf-8")
    except (TypeError, ValueError) as exc: raise ValueError("payload must be JSON-serializable") from exc
    if len(encoded) > max_bytes: raise ValueError("payload exceeds size limit")
    if not isinstance(payload, (dict, list, str, int, float, bool, type(None))):
        raise ValueError("unsupported payload type")
    return payload
