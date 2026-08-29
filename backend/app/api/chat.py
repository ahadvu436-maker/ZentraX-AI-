"""
ZentraX AI — Chat API Router
===============================
Endpoints for sending messages (creating conversations implicitly on first
message) and retrieving conversation history. Every route requires an
authenticated user via `get_current_user`, and every conversation lookup is
scoped to `current_user.id` — one user can never read or write into
another user's conversation, even by guessing a valid conversation UUID.
"""

from __future__ import annotations

import logging
import uuid

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.deps import get_current_user
from app.database.session import get_db_session as get_db
from app.models.conversation import Conversation
from app.models.messages import Message, SenderType
from app.models.user import User
from app.schemas.chat import (
    ConversationHistoryResponse,
    ConversationResponse,
    MessageResponse,
    SendMessageRequest,
    SendMessageResponse,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/chat", tags=["Chat"])


async def _get_owned_conversation(
    conversation_id: uuid.UUID,
    user_id: uuid.UUID,
    db: AsyncSession,
) -> Conversation:
    """
    Fetch a conversation and verify it belongs to `user_id`.

    Raises 404 (never 403) when the conversation doesn't exist OR belongs
    to someone else — the distinction isn't disclosed, so this endpoint
    can't be used to enumerate other users' conversation IDs.
    """
    result = await db.execute(
        select(Conversation).where(
            Conversation.id == conversation_id,
            Conversation.user_id == user_id,
        )
    )
    conversation = result.scalar_one_or_none()
    if conversation is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Conversation not found.",
        )
    return conversation


async def _generate_ai_response(user_message_content: str) -> tuple[str, int | None]:
    """
    Produce the assistant's reply to a user message.

    Placeholder — swap this for a real call to your LLM inference layer
    (e.g. an internal `app.services.llm` module). Kept as an isolated,
    single-purpose async function so that integration is a one-place
    change and doesn't touch the request/response/persistence logic below.

    Returns:
        (reply_content, token_usage)
    """
    # TODO: replace with real model inference.
    reply = (
        "This is a placeholder response. Connect `_generate_ai_response` "
        "to your LLM inference service to return real completions."
    )
    return reply, None


@router.post(
    "/messages",
    response_model=SendMessageResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Send a message, creating a new conversation if none is specified",
)
async def send_message(
    payload: SendMessageRequest,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> SendMessageResponse:
    """
    Append a user message to a conversation (creating one if
    `conversation_id` is omitted), generate the assistant's reply, and
    persist both turns.
    """
    if payload.conversation_id is not None:
        conversation = await _get_owned_conversation(
            payload.conversation_id, current_user.id, db
        )
    else:
        conversation = Conversation(user_id=current_user.id)
        db.add(conversation)
        await db.flush()  # populate conversation.id for the messages below

    user_message = Message(
        conversation_id=conversation.id,
        sender_type=SenderType.USER,
        content=payload.content,
    )
    db.add(user_message)

    try:
        ai_content, ai_token_usage = await _generate_ai_response(payload.content)
    except Exception:
        logger.exception(
            "AI response generation failed for conversation_id=%s", conversation.id
        )
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="Failed to generate a response. Please try again.",
        )

    ai_message = Message(
        conversation_id=conversation.id,
        sender_type=SenderType.AI,
        content=ai_content,
        token_usage=ai_token_usage,
    )
    db.add(ai_message)

    await db.flush()
    await db.refresh(user_message)
    await db.refresh(ai_message)

    logger.info(
        "Message exchange recorded: conversation_id=%s user_id=%s",
        conversation.id,
        current_user.id,
    )

    return SendMessageResponse(
        conversation_id=conversation.id,
        user_message=MessageResponse.model_validate(user_message),
        ai_message=MessageResponse.model_validate(ai_message),
    )


@router.get(
    "/conversations/{conversation_id}/messages",
    response_model=ConversationHistoryResponse,
    summary="Retrieve message history for a conversation",
)
async def get_conversation_history(
    conversation_id: uuid.UUID,
    limit: int = Query(default=100, ge=1, le=500),
    offset: int = Query(default=0, ge=0),
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> ConversationHistoryResponse:
    """
    Return a conversation's metadata plus its messages, oldest first,
    paginated via `limit`/`offset`.
    """
    conversation = await _get_owned_conversation(conversation_id, current_user.id, db)

    result = await db.execute(
        select(Message)
        .where(Message.conversation_id == conversation.id)
        .order_by(Message.created_at.asc())
        .offset(offset)
        .limit(limit)
    )
    messages = result.scalars().all()

    return ConversationHistoryResponse(
        conversation=ConversationResponse.model_validate(conversation),
        messages=[MessageResponse.model_validate(m) for m in messages],
    )