"""Asynchronous message bus for agent-to-agent communication.

Supports direct messages, request/response with timeouts, pub/sub topics
(with ``fnmatch`` wildcards such as ``"research.*"``), broadcast and a bounded
dead-letter queue for undeliverable messages.
"""

from __future__ import annotations

import asyncio
import fnmatch
import logging
from collections import defaultdict, deque
from typing import Any, Deque, Dict, List, Optional, Set, Tuple

from .agent_core import (
    BaseAgent,
    CommunicationError,
    Message,
    MessageType,
)

logger = logging.getLogger("zentrax.agents.bus")

__all__ = ["MessageBus", "Message", "MessageType"]


class MessageBus:
    """In-process async message transport. Implements ``MessageTransport``."""

    def __init__(self, *, dead_letter_limit: int = 1000) -> None:
        self._agents: Dict[str, BaseAgent] = {}
        self._subscriptions: Dict[str, Set[str]] = defaultdict(set)  # pattern -> agent ids
        self._pending: Dict[str, Tuple["asyncio.Future[Message]", str]] = {}  # request id -> (future, recipient)
        self.dead_letters: Deque[Message] = deque(maxlen=dead_letter_limit)
        self.stats: Dict[str, int] = {"sent": 0, "delivered": 0, "dead_lettered": 0}

    # ------------------------------------------------------------------ #
    # Membership
    # ------------------------------------------------------------------ #

    @property
    def agent_ids(self) -> List[str]:
        return list(self._agents)

    def __contains__(self, agent_id: object) -> bool:
        return agent_id in self._agents

    def register(self, agent: BaseAgent) -> None:
        existing = self._agents.get(agent.id)
        if existing is not None and existing is not agent:
            raise CommunicationError(f"agent id {agent.id!r} is already registered on the bus")
        self._agents[agent.id] = agent
        agent.attach_transport(self)

    def unregister(self, agent_id: str) -> None:
        agent = self._agents.pop(agent_id, None)
        if agent is not None:
            agent.attach_transport(None)
        for subscribers in self._subscriptions.values():
            subscribers.discard(agent_id)
        for request_id, (future, recipient) in list(self._pending.items()):
            if recipient == agent_id and not future.done():
                future.set_exception(CommunicationError(f"agent {agent_id} left the bus"))

    def subscribe(self, agent_id: str, topic_pattern: str) -> None:
        if agent_id not in self._agents:
            raise CommunicationError(f"unknown agent {agent_id!r}")
        self._subscriptions[topic_pattern].add(agent_id)

    def unsubscribe(self, agent_id: str, topic_pattern: Optional[str] = None) -> None:
        patterns = [topic_pattern] if topic_pattern else list(self._subscriptions)
        for pattern in patterns:
            self._subscriptions[pattern].discard(agent_id)

    # ------------------------------------------------------------------ #
    # Sending
    # ------------------------------------------------------------------ #

    async def send(self, message: Message) -> bool:
        """Route ``message``. Returns ``True`` if it reached at least one agent."""
        self.stats["sent"] += 1

        if message.type in (MessageType.RESPONSE, MessageType.ERROR) and message.correlation_id:
            entry = self._pending.get(message.correlation_id)
            if entry is None:
                logger.debug("dropping late reply %s", message.id)
                return False
            future, _ = entry
            if not future.done():
                future.set_result(message)
            return True

        if message.recipient is None:
            targets = [aid for aid in self._agents if aid != message.sender]
            return await self._fan_out(message, targets) > 0

        target = self._agents.get(message.recipient)
        if target is None:
            self._dead_letter(message, f"unknown recipient {message.recipient!r}")
            return False
        return await self._deliver(target, message)

    async def request(
        self, sender_id: str, recipient_id: str, payload: Dict[str, Any], timeout: float = 30.0
    ) -> Message:
        """Send a REQUEST and wait for the matching RESPONSE."""
        message = Message(sender_id, recipient_id, dict(payload), MessageType.REQUEST)
        future: "asyncio.Future[Message]" = asyncio.get_running_loop().create_future()
        self._pending[message.id] = (future, recipient_id)
        try:
            await self.send(message)
            try:
                response = await asyncio.wait_for(future, timeout=timeout)
            except asyncio.TimeoutError:
                raise CommunicationError(
                    f"request {message.id} to {recipient_id} timed out after {timeout}s"
                ) from None
        finally:
            self._pending.pop(message.id, None)
        if response.type == MessageType.ERROR:
            raise CommunicationError(str(response.payload.get("error", "remote error")))
        return response

    async def publish(self, sender_id: str, topic: str, payload: Dict[str, Any]) -> int:
        """Deliver an EVENT to every subscriber whose pattern matches ``topic``."""
        recipients: Set[str] = set()
        for pattern, subscribers in self._subscriptions.items():
            if subscribers and fnmatch.fnmatchcase(topic, pattern):
                recipients |= subscribers
        recipients.discard(sender_id)
        delivered = 0
        for recipient in recipients:
            target = self._agents.get(recipient)
            if target is None:
                continue
            message = Message(sender_id, recipient, dict(payload), MessageType.EVENT, topic)
            self.stats["sent"] += 1
            if await self._deliver(target, message):
                delivered += 1
        return delivered

    async def broadcast(self, sender_id: str, payload: Dict[str, Any]) -> int:
        message = Message(sender_id, None, dict(payload), MessageType.BROADCAST)
        self.stats["sent"] += 1
        targets = [aid for aid in self._agents if aid != sender_id]
        return await self._fan_out(message, targets)

    # ------------------------------------------------------------------ #
    # Internals
    # ------------------------------------------------------------------ #

    async def _fan_out(self, message: Message, targets: List[str]) -> int:
        delivered = 0
        for agent_id in targets:
            target = self._agents.get(agent_id)
            if target is None:
                continue
            copy = Message(
                message.sender, agent_id, dict(message.payload), message.type, message.topic, message.correlation_id
            )
            if await self._deliver(target, copy):
                delivered += 1
        return delivered

    async def _deliver(self, target: BaseAgent, message: Message) -> bool:
        try:
            await target.deliver(message)
        except CommunicationError as exc:
            self._dead_letter(message, str(exc))
            return False
        self.stats["delivered"] += 1
        return True

    def _dead_letter(self, message: Message, reason: str) -> None:
        self.stats["dead_lettered"] += 1
        self.dead_letters.append(message)
        logger.warning("dead letter %s -> %s: %s", message.sender, message.recipient, reason)
        if message.type == MessageType.REQUEST:
            entry = self._pending.get(message.id)
            if entry is not None and not entry[0].done():
                entry[0].set_exception(CommunicationError(reason))
 
