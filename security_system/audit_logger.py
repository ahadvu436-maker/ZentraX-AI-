import json, os, threading, time
class AuditLogger:
    def __init__(self, path=None):
        self.path, self._lock = path, threading.Lock()
    def write(self, event, fields):
        record = {"time": time.time(), "event": event, "fields": self._redact(fields)}
        line = json.dumps(record, sort_keys=True, default=str)
        if self.path:
            os.makedirs(os.path.dirname(os.path.abspath(self.path)), exist_ok=True)
            with self._lock, open(self.path, "a", encoding="utf-8") as f:
                f.write(line + "\\n"); f.flush(); os.fsync(f.fileno())
        return record
    @classmethod
    def _redact(cls, value):
        if isinstance(value, dict):
            return {k: ("[REDACTED]" if any(s in k.lower() for s in ("password","token","secret","key")) else cls._redact(v)) for k,v in value.items()}
        if isinstance(value, list): return [cls._redact(v) for v in value]
        return value
