"""
File: chat.py
Description: FastAPI router for the chatbot endpoints.
Dependencies: fastapi, src.services.chat_service
Author: FlavorFlow Team

Provides REST endpoints that the React frontend will consume:

  POST /chat/configure     — set API key & provider
  POST /chat/message       — send a message, get a reply
  POST /chat/stream        — SSE stream of tokens
  GET  /chat/history       — get conversation history
  POST /chat/reset         — clear conversation
  GET  /chat/sessions      — list all sessions
  DELETE /chat/sessions/{id} — delete a session
  GET  /chat/health        — LLM connection health check
  GET  /chat/config        — current config status
  GET  /chat/usage         — token usage stats
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from typing import Any, Dict, List, Optional
import json

from src.services.chat_service import ChatService
from src.services.llm_service import LLMError


# ─── Pydantic schemas ────────────────────────────────────────────────────────

class ConfigureRequest(BaseModel):
    """Request to configure the LLM provider."""
    api_key: Optional[str] = Field(
        None, description="LLM provider API key (optional if already configured)"
    )
    provider: str = Field(
        "groq",
        description="Provider: 'groq' (default), 'gemini', or 'openai_compatible'",
    )
    model: Optional[str] = Field(
        None, description="Model override (uses provider default if omitted)"
    )
    base_url: Optional[str] = Field(
        None, description="Base URL override for custom endpoints"
    )
    temperature: Optional[float] = Field(
        None, ge=0, le=2, description="Sampling temperature (0-2)"
    )
    max_tokens: Optional[int] = Field(
        None, ge=1, le=8192, description="Max response tokens"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "api_key": "gsk_your_groq_key_here",
                "provider": "groq",
                "model": "llama-3.3-70b-versatile",
            }
        }


class ChatMessageRequest(BaseModel):
    """Request to send a chat message."""
    message: str = Field(..., min_length=1, description="User message text")
    session_id: Optional[str] = Field(
        None, description="Session ID (auto-created if omitted)"
    )
    temperature: Optional[float] = Field(None, ge=0, le=2)
    max_tokens: Optional[int] = Field(None, ge=1, le=8192)

    class Config:
        json_schema_extra = {
            "example": {
                "message": "What are my most critical inventory items?",
                "session_id": None,
            }
        }


class ChatMessageResponse(BaseModel):
    """Response from the chatbot."""
    session_id: str
    reply: str
    usage: Dict[str, Any] = {}
    model: str = ""
    provider: str = ""


class ResetRequest(BaseModel):
    """Request to reset a session."""
    session_id: str


class ConfigStatusResponse(BaseModel):
    """Current LLM configuration status."""
    is_configured: bool
    is_context_loaded: bool
    provider: str
    model: str
    base_url: str
    temperature: float
    max_tokens: int
    api_key: Optional[str] = None
    active_sessions: int = 0


class HealthResponse(BaseModel):
    """LLM health check result."""
    ok: bool
    provider: str
    model: str
    error: Optional[str] = None


# ─── Singleton chat service ──────────────────────────────────────────────────

_chat_service: Optional[ChatService] = None


def get_chat_service() -> ChatService:
    """Get or lazily create the global ChatService instance."""
    global _chat_service
    if _chat_service is None:
        _chat_service = ChatService()  # unconfigured until /chat/configure
    return _chat_service


def set_chat_service(service: ChatService) -> None:
    """Replace the global ChatService (used by main.py on startup)."""
    global _chat_service
    _chat_service = service


# ─── Router ──────────────────────────────────────────────────────────────────

router = APIRouter(prefix="/chat", tags=["Chat / AI Assistant"])


@router.post(
    "/configure",
    response_model=ConfigStatusResponse,
    summary="Configure the LLM provider",
)
async def configure_llm(req: ConfigureRequest):
    """
    Set the API key and provider for the chatbot.

    Must be called once before sending messages.
    Supports: **groq** (free, default), **gemini**, **openai_compatible**.
    """
    svc = get_chat_service()
    result = svc.configure(
        api_key=req.api_key,
        provider=req.provider,
        model=req.model or "",
        base_url=req.base_url or "",
        temperature=req.temperature,
        max_tokens=req.max_tokens,
    )
    return ConfigStatusResponse(**result)


@router.get(
    "/config",
    response_model=ConfigStatusResponse,
    summary="Get current LLM config status",
)
async def get_config():
    """Returns the current configuration (API key is masked)."""
    svc = get_chat_service()
    return ConfigStatusResponse(**svc.get_config_status())


@router.post(
    "/message",
    response_model=ChatMessageResponse,
    summary="Send a chat message",
)
async def send_message(req: ChatMessageRequest):
    """
    Send a user message and receive the assistant's reply.

    A new session is automatically created if ``session_id`` is omitted.
    The assistant has full context from the latest data analysis.
    """
    svc = get_chat_service()
    if not svc.is_configured:
        raise HTTPException(
            status_code=400,
            detail="LLM not configured. Call POST /chat/configure first.",
        )
    try:
        result = await svc.send_message(
            message=req.message,
            session_id=req.session_id,
            temperature=req.temperature,
            max_tokens=req.max_tokens,
        )
        return ChatMessageResponse(**result)
    except LLMError as exc:
        raise HTTPException(status_code=502, detail=str(exc))
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc))


@router.post("/stream", summary="Stream a chat response (SSE)")
async def stream_message(req: ChatMessageRequest):
    """
    Stream the assistant's response token-by-token via Server-Sent Events.

    Each SSE event is a JSON object: ``{"token": "..."}``
    The final event is: ``{"done": true, "session_id": "..."}``
    """
    svc = get_chat_service()
    if not svc.is_configured:
        raise HTTPException(
            status_code=400,
            detail="LLM not configured. Call POST /chat/configure first.",
        )

    sid = svc.get_or_create_session(req.session_id)

    async def event_generator():
        try:
            async for token in svc.stream_message(
                message=req.message,
                session_id=sid,
                temperature=req.temperature,
                max_tokens=req.max_tokens,
            ):
                yield f"data: {json.dumps({'token': token})}\n\n"
            yield f"data: {json.dumps({'done': True, 'session_id': sid})}\n\n"
        except LLMError as exc:
            yield f"data: {json.dumps({'error': str(exc)})}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@router.get("/history", summary="Get conversation history")
async def get_history(
    session_id: str = Query(..., description="Session ID"),
    include_system: bool = Query(False, description="Include system prompt"),
):
    """Return the message history for a session."""
    svc = get_chat_service()
    try:
        return svc.get_history(session_id, include_system=include_system)
    except KeyError:
        raise HTTPException(status_code=404, detail="Session not found")


@router.post("/reset", summary="Reset a conversation")
async def reset_session(req: ResetRequest):
    """Clear the conversation history, keeping the session alive."""
    svc = get_chat_service()
    try:
        svc.reset_session(req.session_id)
        return {"status": "reset", "session_id": req.session_id}
    except KeyError:
        raise HTTPException(status_code=404, detail="Session not found")


@router.get("/sessions", summary="List all chat sessions")
async def list_sessions():
    """Return a list of all active chat sessions."""
    svc = get_chat_service()
    return svc.list_sessions()


@router.delete("/sessions/{session_id}", summary="Delete a session")
async def delete_session(session_id: str):
    """Permanently delete a chat session and its history."""
    svc = get_chat_service()
    if svc.delete_session(session_id):
        return {"status": "deleted", "session_id": session_id}
    raise HTTPException(status_code=404, detail="Session not found")


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="LLM health check",
)
async def llm_health():
    """Quick check that the LLM provider is reachable and the key is valid."""
    svc = get_chat_service()
    if not svc.is_configured:
        return HealthResponse(
            ok=False,
            provider=svc.provider,
            model=svc.model,
            error="Not configured",
        )
    result = await svc.health_check()
    return HealthResponse(**result)


@router.get("/usage", summary="Token usage statistics")
async def get_usage():
    """Return cumulative token usage across all calls in this session."""
    svc = get_chat_service()
    return svc.get_usage_stats()
