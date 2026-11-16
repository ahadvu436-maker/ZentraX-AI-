"""Wire protocol: envelope model, validation, JSON codec and optional HMAC signing."""
from __future__ import annotations

import enum
import hashlib
import hmac
import json
import time
import uuid
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Dict, Mapping, Optional, Protocol, Union

from .communication_core import CommunicationError, ProtocolError

PROTOCOL_VERSION = 1
_MAX_ID_LEN = 128
_MAX_TOPIC_LEN = 256


class MessageType(str, enum.Enum):
    REQUEST = "request"
    RESPONSE = "response"
    EVENT = "event"
    ERROR = "error"
    PING = "ping"
    PONG = "pong"
    SUBSCRIBE = "subscribe"
    UNSUBSCRIBE = "unsubscribe"
    ACK = "ack"


@dataclass(frozen=True, eq=False)
class Envelope:
    """Transport-independent message. ``correlation_id`` links replies to requests."""

    type: MessageType
    topic: str = ""
    payload: Any = None
    id: str = field(default_factory=lambda: uuid.uuid4().hex)
    correlation_id: Optional[str] = None
    reply_to: Optional[str] = None
    timestamp: float = field(default_factory=time.time)
    version: int = PROTOCOL_VERSION
    headers: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "headers", MappingProxyType(dict(self.headers)))

    def to_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {"v": self.version, "id": self.id, "type": self.type.value, "ts": self.timestamp}
        if self.topic:
            d["topic"] = self.topic
        if self.payload is not None:
            d["payload"] = self.payload
        if self.correlation_id:
            d["cid"] = self.correlation_id
        if self.reply_to:
            d["reply_to"] = self.reply_to
        if self.headers:
            d["headers"] = dict(self.headers)
        return d


class Codec(Protocol):
    """Text codec. Implementations must raise ``ValueError``/``TypeError`` on bad data."""

    def encode(self, data: Mapping[str, Any]) -> str: ...
    def decode(self, raw: Union[str, bytes]) -> Any: ...


class JsonCodec:
    """Strict JSON: no NaN/Infinity, compact separators, UTF-8 preserved."""

    def encode(self, data: Mapping[str, Any]) -> str:
        return json.dumps(data, separators=(",", ":"), ensure_ascii=False, allow_nan=False)

    def decode(self, raw: Union[str, bytes]) -> Any:
        return json.loads(raw, parse_constant=_reject_constant)


def _reject_constant(name: str) -> Any:
    raise ValueError(f"invalid JSON constant {name}")


class ProtocolHandler:
    """Encodes/decodes/validates envelopes. With ``secret`` set, every frame is
    signed (HMAC-SHA256) and unsigned/tampered frames are rejected."""

    def __init__(self, codec: Optional[Codec] = None, *, secret: Optional[bytes] = None,
                 max_message_bytes: int = 1_048_576, supported_versions: frozenset = frozenset({PROTOCOL_VERSION})) -> None:
        if max_message_bytes <= 0:
            raise ValueError("max_message_bytes must be > 0")
        self._codec: Codec = codec or JsonCodec()
        self._secret = secret
        self._max = max_message_bytes
        self._versions = supported_versions

    # -- construction helpers ---------------------------------------------- #
    def make(self, type: MessageType, *, topic: str = "", payload: Any = None,
             correlation_id: Optional[str] = None, reply_to: Optional[str] = None,
             headers: Optional[Mapping[str, str]] = None) -> Envelope:
        return Envelope(type=type, topic=topic, payload=payload, correlation_id=correlation_id,
                        reply_to=reply_to, headers=headers or {})

    def make_error(self, request: Optional[Envelope], exc: BaseException) -> Envelope:
        """Build an ERROR envelope. Non-protocol exceptions are masked to avoid leaking internals."""
        if isinstance(exc, CommunicationError):
            payload = {"code": exc.code, "message": str(exc), "status": exc.status}
        else:
            payload = {"code": "internal_error", "message": "internal error", "status": 500}
        return self.make(MessageType.ERROR, topic=request.topic if request else "", payload=payload,
                         correlation_id=request.id if request else None)

    # -- wire format -------------------------------------------------------- #
    def encode(self, env: Envelope) -> str:
        data = env.to_dict()
        if self._secret:
            data["sig"] = self._sign(data)
        try:
            text = self._codec.encode(data)
        except (TypeError, ValueError, RecursionError) as exc:
            raise ProtocolError("payload is not serialisable", code="encode_error") from exc
        if len(text.encode("utf-8")) > self._max:
            raise ProtocolError("message too large", code="message_too_large", status=413)
        return text

    def decode(self, raw: Union[str, bytes]) -> Envelope:
        if len(raw) > self._max:
            raise ProtocolError("message too large", code="message_too_large", status=413)
        try:
            data = self._codec.decode(raw)
        except (ValueError, TypeError, RecursionError) as exc:
            raise ProtocolError("malformed message", code="malformed") from exc
        if not isinstance(data, dict):
            raise ProtocolError("message must be an object", code="malformed")
        if self._secret:
            sig = data.pop("sig", None)
            if not isinstance(sig, str) or not hmac.compare_digest(sig, self._safe_sign(data)):
                raise ProtocolError("invalid signature", code="bad_signature", status=401)
        return self._validate(data)

    # -- internals ---------------------------------------------------------- #
    def _canonical(self, data: Mapping[str, Any]) -> bytes:
        return json.dumps(data, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
                          allow_nan=False).encode("utf-8")

    def _sign(self, data: Mapping[str, Any]) -> str:
        assert self._secret is not None
        return hmac.new(self._secret, self._canonical(data), hashlib.sha256).hexdigest()

    def _safe_sign(self, data: Mapping[str, Any]) -> str:
        try:
            return self._sign(data)
        except (TypeError, ValueError, RecursionError) as exc:
            raise ProtocolError("malformed message", code="malformed") from exc

    def _validate(self, d: Dict[str, Any]) -> Envelope:
        try:
            mtype = MessageType(d.get("type"))
        except ValueError:
            raise ProtocolError("unknown message type", code="unknown_type") from None
        version = d.get("v", PROTOCOL_VERSION)
        if not isinstance(version, int) or isinstance(version, bool) or version not in self._versions:
            raise ProtocolError("unsupported protocol version", code="unsupported_version")
        msg_id = d.get("id") or uuid.uuid4().hex
        if not isinstance(msg_id, str) or len(msg_id) > _MAX_ID_LEN:
            raise ProtocolError("invalid id", code="invalid_field")
        topic = d.get("topic", "")
        if not isinstance(topic, str) or len(topic) > _MAX_TOPIC_LEN:
            raise ProtocolError("invalid topic", code="invalid_field")
        if mtype in (MessageType.REQUEST, MessageType.EVENT) and not topic:
            raise ProtocolError("topic is required", code="missing_topic")
        for key in ("cid", "reply_to"):
            v = d.get(key)
            if v is not None and (not isinstance(v, str) or len(v) > _MAX_ID_LEN):
                raise ProtocolError(f"invalid {key}", code="invalid_field")
        headers = d.get("headers", {})
        if not isinstance(headers, dict) or not all(isinstance(k, str) and isinstance(v, str) for k, v in headers.items()):
            raise ProtocolError("headers must be a string map", code="invalid_field")
        ts = d.get("ts", time.time())
        if not isinstance(ts, (int, float)) or isinstance(ts, bool):
            raise ProtocolError("invalid timestamp", code="invalid_field")
        return Envelope(type=mtype, topic=topic, payload=d.get("payload"), id=msg_id,
                        correlation_id=d.get("cid"), reply_to=d.get("reply_to"),
                        timestamp=float(ts), version=version, headers=headers)
