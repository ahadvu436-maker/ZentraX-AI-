"""
chat_service.py

Core conversation service for ZentraX AI.

Responsibilities:
    - Own the lifecycle of a chat session (create, append, expire).
    - Validate and sanitize inbound messages before they touch a model.
    - Dispatch messages to a pluggable AI provider (local model or external API)
      via an abstract interface, so providers can be swapped without touching
      business logic.
    - Enforce privacy defaults: no raw message content in logs, configurable
      history retention, and explicit redaction hooks.

This module intentionally contains NO provider-specific HTTP code — see
`app/services/providers/` for concrete implementations of `AIProvider`.
Keeping that boundary is what lets this file stay provider-agnostic.
"""

from __future__ import annotations

import asyncio
import logging
import re
import time
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Protocol

logger = logging.getLogger("zentrax.chat_service")

# -----------------------------------------------------------------------------
# Configuration constants
# -----------------------------------------------------------------------------

MAX_MESSAGE_LENGTH = 8_000          # chars; guards against abuse / runaway cost
MAX_HISTORY_MESSAGES = 40           # rolling window sent to the model
SESSION_TTL_SECONDS = 60 * 60 * 2   # 2 hours of inactivity -> session expires
PROVIDER_TIMEOUT_SECONDS = 30
PROVIDER_MAX_RETRIES = 2

# Very small denylist of patterns we never want to accidentally log verbatim.
# This is NOT a security control on its own — it's a defense-in-depth guard
# against secrets/PII leaking into application logs.
_SENSITIVE_PATTERNS = [
    re.compile(r"\b\d{13,19}\b"),                       # card-like number runs
    re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b"),         # email addresses
    re.compile(r"sk-[A-Za-z0-9]{16,}"),                  # API-key-shaped tokens
]


class Role(str, Enum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"


# -----------------------------------------------------------------------------
# Data models
# -----------------------------------------------------------------------------

@dataclass(slots=True)
class ChatMessage:
    """A single turn in a conversation."""
    role: Role
    content: str
    created_at: float = field(default_factory=time.time)


@dataclass(slots=True)
class ChatSession:
    """
    In-memory representation of a conversation.

    `user_ref` deliberately stores an opaque identifier (e.g. a hashed user id
    or session token), never raw PII like email/name, so a dump of session
    state does not itself become a privacy liability.
    """
    session_id: str
    user_ref: str
    messages: list[ChatMessage] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)
    last_active_at: float = field(default_factory=time.time)

    def is_expired(self, ttl_seconds: int = SESSION_TTL_SECONDS) -> bool:
        return (time.time() - self.last_active_at) > ttl_seconds

    def touch(self) -> None:
        self.last_active_at = time.time()


class ChatServiceError(Exception):
    """Base class for user-facing chat service errors."""


class MessageTooLongError(ChatServiceError):
    pass


class SessionNotFoundError(ChatServiceError):
    pass


class ProviderUnavailableError(ChatServiceError):
    """Raised when the upstream model/provider fails after retries."""


# -----------------------------------------------------------------------------
# Provider abstraction
# -----------------------------------------------------------------------------

class AIProvider(Protocol):
    """
    Minimal interface any model backend must satisfy — a local model runner,
    or a wrapper around an external AI API. Keeping this a `Protocol` (rather
    than requiring inheritance) makes it trivial to adapt existing clients.
    """

    async def generate(
        self,
        messages: list[ChatMessage],
        *,
        timeout: float = PROVIDER_TIMEOUT_SECONDS,
    ) -> str:
        """Return the assistant's reply text for the given message history."""
        ...


class SessionStore(ABC):
    """
    Storage interface for sessions. Ship an in-memory implementation for dev
    and tests; swap in a Redis/DB-backed implementation for production so
    sessions survive restarts and work across multiple backend instances.
    """

    @abstractmethod
    async def get(self, session_id: str) -> Optional[ChatSession]: ...

    @abstractmethod
    async def save(self, session: ChatSession) -> None: ...

    @abstractmethod
    async def delete(self, session_id: str) -> None: ...


class InMemorySessionStore(SessionStore):
    """Simple asyncio-safe in-memory store. Fine for local dev; not durable
    and not shared across processes — do not use as-is in multi-instance prod."""

    def __init__(self) -> None:
        self._sessions: dict[str, ChatSession] = {}
        self._lock = asyncio.Lock()

    async def get(self, session_id: str) -> Optional[ChatSession]:
        async with self._lock:
            return self._sessions.get(session_id)

    async def save(self, session: ChatSession) -> None:
        async with self._lock:
            self._sessions[session.session_id] = session

    async def delete(self, session_id: str) -> None:
        async with self._lock:
            self._sessions.pop(session_id, None)


# -----------------------------------------------------------------------------
# Chat service
# -----------------------------------------------------------------------------

class ChatService:
    """
    Orchestrates session state and model calls for a conversation.

    Usage:
        service = ChatService(provider=my_provider, store=InMemorySessionStore())
        session_id = await service.start_session(user_ref="hashed-user-id")
        reply = await service.send_message(session_id, "Hello!")
    """

    def __init__(
        self,
        provider: AIProvider,
        store: Optional[SessionStore] = None,
        *,
        system_prompt: Optional[str] = None,
        max_history_messages: int = MAX_HISTORY_MESSAGES,
    ) -> None:
        self._provider = provider
        self._store = store or InMemorySessionStore()
        self._system_prompt = system_prompt
        self._max_history_messages = max_history_messages

    # -- Session lifecycle ----------------------------------------------------

    async def start_session(self, user_ref: str) -> str:
        """Create a new session for the given (already-anonymized) user reference."""
        session = ChatSession(session_id=str(uuid.uuid4()), user_ref=user_ref)
        if self._system_prompt:
            session.messages.append(ChatMessage(Role.SYSTEM, self._system_prompt))
        await self._store.save(session)
        logger.info("session_started session_id=%s", session.session_id)
        return session.session_id

    async def end_session(self, session_id: str) -> None:
        """Explicitly terminate and discard a session (user-initiated 'clear chat')."""
        await self._store.delete(session_id)
        logger.info("session_ended session_id=%s", session_id)

    async def _get_active_session(self, session_id: str) -> ChatSession:
        session = await self._store.get(session_id)
        if session is None:
            raise SessionNotFoundError(f"No session found for id={session_id}")
        if session.is_expired():
            await self._store.delete(session_id)
            raise SessionNotFoundError(f"Session {session_id} expired")
        return session

    # -- Message handling -------------------------------------------------------

    async def send_message(self, session_id: str, content: str) -> str:
        """
        Validate, store, and send a user message to the model; store and
        return the assistant's reply.

        Raises:
            MessageTooLongError, SessionNotFoundError, ProviderUnavailableError
        """
        content = self._validate_and_sanitize(content)

        session = await self._get_active_session(session_id)
        session.messages.append(ChatMessage(Role.USER, content))
        session.touch()

        history = self._trim_history(session.messages)

        try:
            reply_text = await self._call_provider_with_retries(history)
        except Exception as exc:  # noqa: BLE001 - normalize all provider failures
            logger.error(
                "provider_call_failed session_id=%s error_type=%s",
                session_id, type(exc).__name__,
            )
            raise ProviderUnavailableError("The AI provider is currently unavailable.") from exc

        session.messages.append(ChatMessage(Role.ASSISTANT, reply_text))
        session.touch()
        await self._store.save(session)

        # Privacy: log metadata only (lengths, ids) — never message content.
        logger.info(
            "message_processed session_id=%s in_chars=%d out_chars=%d",
            session_id, len(content), len(reply_text),
        )
        return reply_text

    async def get_history(self, session_id: str, *, include_system: bool = False) -> list[ChatMessage]:
        """Return the visible conversation history for a session."""
        session = await self._get_active_session(session_id)
        if include_system:
            return list(session.messages)
        return [m for m in session.messages if m.role != Role.SYSTEM]

    # -- Internal helpers ---------------------------------------------------------

    def _validate_and_sanitize(self, content: str) -> str:
        if not content or not content.strip():
            raise ChatServiceError("Message content must not be empty.")

        content = content.strip()

        if len(content) > MAX_MESSAGE_LENGTH:
            raise MessageTooLongError(
                f"Message exceeds max length of {MAX_MESSAGE_LENGTH} characters."
            )

        # Strip control characters (except newline/tab) to prevent log/UI injection.
        content = "".join(ch for ch in content if ch == "\n" or ch == "\t" or ch.isprintable())

        return content

    def _trim_history(self, messages: list[ChatMessage]) -> list[ChatMessage]:
        """Keep the system prompt (if present) plus the most recent N turns,
        so context stays bounded regardless of how long a session runs."""
        if not messages:
            return messages
        system_msgs = [m for m in messages if m.role == Role.SYSTEM]
        convo_msgs = [m for m in messages if m.role != Role.SYSTEM]
        return system_msgs + convo_msgs[-self._max_history_messages:]

    async def _call_provider_with_retries(self, history: list[ChatMessage]) -> str:
        last_exc: Optional[Exception] = None
        for attempt in range(1, PROVIDER_MAX_RETRIES + 2):
            try:
                return await asyncio.wait_for(
                    self._provider.generate(history, timeout=PROVIDER_TIMEOUT_SECONDS),
                    timeout=PROVIDER_TIMEOUT_SECONDS,
                )
            except Exception as exc:  # noqa: BLE001
                last_exc = exc
                logger.warning(
                    "provider_attempt_failed attempt=%d/%d error_type=%s",
                    attempt, PROVIDER_MAX_RETRIES + 1, type(exc).__name__,
                )
                if attempt <= PROVIDER_MAX_RETRIES:
                    await asyncio.sleep(min(2 ** attempt, 8))  # simple backoff
        assert last_exc is not None
        raise last_exc


def redact_for_logging(text: str) -> str:
    """
    Best-effort redaction helper for any code path that must log a snippet
    of message content (e.g. debugging). Not used by default logging above,
    which avoids content entirely — reach for this only when content-aware
    logging is genuinely necessary.
    """
    redacted = text
    for pattern in _SENSITIVE_PATTERNS:
        redacted = pattern.sub("[REDACTED]", redacted)
    return redacted