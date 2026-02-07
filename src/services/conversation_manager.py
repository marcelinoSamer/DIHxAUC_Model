"""
File: conversation_manager.py
Description: In-memory conversation state management for the chatbot.
Dependencies: (stdlib only)
Author: FlavorFlow Team

Manages chat sessions, message history, and system prompt injection.
Each session is identified by a ``session_id`` string.

Thread-safe for use inside a FastAPI async context.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


# ─── data types ──────────────────────────────────────────────────────────────

@dataclass
class ChatMessage:
    """A single message in a conversation."""
    role: str          # "system" | "user" | "assistant"
    content: str
    timestamp: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_llm_format(self) -> Dict[str, str]:
        """Convert to the {role, content} dict the LLM expects."""
        return {"role": self.role, "content": self.content}


@dataclass
class ChatSession:
    """A conversation session with its full history."""
    session_id: str
    created_at: float = field(default_factory=time.time)
    messages: List[ChatMessage] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def message_count(self) -> int:
        return len(self.messages)

    @property
    def last_activity(self) -> float:
        if self.messages:
            return self.messages[-1].timestamp
        return self.created_at

    def get_llm_messages(self) -> List[Dict[str, str]]:
        """Return messages in the format the LLM service expects."""
        return [m.to_llm_format() for m in self.messages]


# ─── manager ─────────────────────────────────────────────────────────────────

class ConversationManager:
    """
    Manages multiple concurrent chat sessions in memory.

    Features:
    - Automatic system prompt injection at session start.
    - Sliding-window history trimming to stay within token budgets.
    - Session expiry / cleanup.

    Usage:
        >>> mgr = ConversationManager(system_prompt="You are a helpful bot.")
        >>> sid = mgr.create_session()
        >>> mgr.add_user_message(sid, "What is the peak hour?")
        >>> messages = mgr.get_llm_messages(sid)
        >>> mgr.add_assistant_message(sid, "The peak hour is 16:00.")
    """

    DEFAULT_MAX_HISTORY = 50          # messages (including system)
    SESSION_TTL_SECONDS = 3600 * 4    # 4 hours

    def __init__(
        self,
        system_prompt: str = "",
        max_history: int = DEFAULT_MAX_HISTORY,
    ) -> None:
        self._system_prompt = system_prompt
        self._max_history = max_history
        self._sessions: Dict[str, ChatSession] = {}

    # ── system prompt ────────────────────────────────────────────────────

    @property
    def system_prompt(self) -> str:
        return self._system_prompt

    @system_prompt.setter
    def system_prompt(self, value: str) -> None:
        """Update system prompt.  Existing sessions keep their old one."""
        self._system_prompt = value

    def update_system_prompt_for_session(
        self, session_id: str, new_prompt: str
    ) -> None:
        """Replace the system message in an existing session."""
        session = self._get_session(session_id)
        if session.messages and session.messages[0].role == "system":
            session.messages[0].content = new_prompt
        else:
            session.messages.insert(
                0, ChatMessage(role="system", content=new_prompt)
            )

    # ── session lifecycle ────────────────────────────────────────────────

    def create_session(
        self,
        session_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Create a new chat session.

        Returns:
            The session ID (auto-generated if not provided).
        """
        sid = session_id or uuid.uuid4().hex[:12]
        session = ChatSession(
            session_id=sid,
            metadata=metadata or {},
        )
        # Inject system prompt as the first message
        if self._system_prompt:
            session.messages.append(
                ChatMessage(role="system", content=self._system_prompt)
            )
        self._sessions[sid] = session
        return sid

    def get_session(self, session_id: str) -> Optional[ChatSession]:
        """Return a session or ``None`` if it doesn't exist."""
        return self._sessions.get(session_id)

    def delete_session(self, session_id: str) -> bool:
        """Delete a session. Returns True if it existed."""
        return self._sessions.pop(session_id, None) is not None

    def list_sessions(self) -> List[Dict[str, Any]]:
        """Return lightweight info about all active sessions."""
        return [
            {
                "session_id": s.session_id,
                "created_at": s.created_at,
                "message_count": s.message_count,
                "last_activity": s.last_activity,
            }
            for s in self._sessions.values()
        ]

    # ── messaging ────────────────────────────────────────────────────────

    def add_user_message(
        self,
        session_id: str,
        content: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> ChatMessage:
        """Append a user message to the session."""
        session = self._get_session(session_id)
        msg = ChatMessage(
            role="user", content=content, metadata=metadata or {}
        )
        session.messages.append(msg)
        self._trim_history(session)
        return msg

    def add_assistant_message(
        self,
        session_id: str,
        content: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> ChatMessage:
        """Append an assistant response to the session."""
        session = self._get_session(session_id)
        msg = ChatMessage(
            role="assistant", content=content, metadata=metadata or {}
        )
        session.messages.append(msg)
        self._trim_history(session)
        return msg

    def get_llm_messages(self, session_id: str) -> List[Dict[str, str]]:
        """Return the message list ready for the LLM service."""
        session = self._get_session(session_id)
        return session.get_llm_messages()

    def get_history(
        self, session_id: str, *, include_system: bool = False
    ) -> List[Dict[str, Any]]:
        """
        Return human-readable history for the frontend.

        Args:
            session_id: Session identifier.
            include_system: Whether to include the system prompt.

        Returns:
            List of ``{role, content, timestamp}`` dicts.
        """
        session = self._get_session(session_id)
        msgs = session.messages
        if not include_system:
            msgs = [m for m in msgs if m.role != "system"]
        return [
            {
                "role": m.role,
                "content": m.content,
                "timestamp": m.timestamp,
            }
            for m in msgs
        ]

    def reset_session(self, session_id: str) -> None:
        """Clear history but keep the session alive with system prompt."""
        session = self._get_session(session_id)
        system_msgs = [m for m in session.messages if m.role == "system"]
        session.messages = system_msgs  # keep system prompt only

    # ── cleanup ──────────────────────────────────────────────────────────

    def cleanup_expired(self) -> int:
        """
        Remove sessions older than SESSION_TTL_SECONDS.

        Returns:
            Number of sessions removed.
        """
        now = time.time()
        expired = [
            sid
            for sid, s in self._sessions.items()
            if now - s.last_activity > self.SESSION_TTL_SECONDS
        ]
        for sid in expired:
            del self._sessions[sid]
        return len(expired)

    # ── internals ────────────────────────────────────────────────────────

    def _get_session(self, session_id: str) -> ChatSession:
        session = self._sessions.get(session_id)
        if session is None:
            raise KeyError(f"Session '{session_id}' not found")
        return session

    def _trim_history(self, session: ChatSession) -> None:
        """Keep messages within ``max_history``, preserving the system prompt."""
        if len(session.messages) <= self._max_history:
            return

        # Always preserve the first message if it's the system prompt
        has_system = (
            session.messages and session.messages[0].role == "system"
        )
        if has_system:
            system = session.messages[0]
            recent = session.messages[-(self._max_history - 1):]
            session.messages = [system] + recent
        else:
            session.messages = session.messages[-self._max_history:]

    @property
    def active_session_count(self) -> int:
        return len(self._sessions)

    def __repr__(self) -> str:
        return (
            f"ConversationManager(sessions={self.active_session_count}, "
            f"max_history={self._max_history})"
        )
