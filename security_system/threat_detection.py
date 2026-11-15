import re
class ThreatDetector:
    """Conservative local heuristics; not a replacement for a WAF or security review."""
    PATTERNS = [re.compile(p, re.I) for p in (r"<script\\b", r"javascript:", r"\\bunion\\s+select\\b", r"\\b(or|and)\\s+1=1\\b")]
    def inspect(self, payload):
        text = payload if isinstance(payload, str) else repr(payload)
        return [p.pattern for p in self.PATTERNS if p.search(text)]
