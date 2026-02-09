"""
File: chat_service.py
Description: High-level chatbot orchestrator that ties everything together.
Dependencies: llm_service, data_context_builder, conversation_manager
Author: FlavorFlow Team

This is the single entry-point the API layer calls.
It manages:
  1. LLM configuration and lifecycle
  2. Data context assembly from analysis results
  3. Conversation state per session
  4. Sending messages and returning structured responses

Usage:
    >>> from src.services.chat_service import ChatService
    >>> chat = ChatService(api_key="gsk_...")
    >>> chat.load_analysis_context(inventory_results, bcg_results)
    >>> reply = await chat.send_message("What items are low on stock?")
"""

from __future__ import annotations

import os
import logging
from typing import Any, AsyncIterator, Dict, List, Optional

from .llm_service import LLMConfig, LLMProvider, LLMService, LLMError
from .data_context_builder import DataContextBuilder
from .conversation_manager import ConversationManager

logger = logging.getLogger("flavorflow.chat")


class ChatService:
    """
    Unified chatbot service.

    Wraps :class:`LLMService`, :class:`DataContextBuilder`, and
    :class:`ConversationManager` into a clean public API.

    Only the API key is required.  Everything else has defaults.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        provider: str = "groq",
        model: str = "",
        base_url: str = "",
        temperature: float = 0.4,
        max_tokens: int = 2048,
    ) -> None:
        # Resolve API key: explicit arg → env var
        resolved_key = api_key or os.getenv("LLM_API_KEY", "")

        self._llm_config = LLMConfig(
            api_key=resolved_key,
            provider=LLMProvider(provider),
            model=model,
            base_url=base_url,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        self._llm = LLMService(self._llm_config)
        self._context_builder = DataContextBuilder()
        # Start with the base persona so the LLM knows its role
        # even before the full data context finishes loading.
        self._conversation_mgr = ConversationManager(
            system_prompt=self._context_builder._PERSONA
            + "\n\nNote: Full data context is still loading. "
            "You can answer general questions but detailed analytics "
            "will be available shortly."
        )

        self._is_context_loaded = False

    # ── configuration ────────────────────────────────────────────────────

    @property
    def is_configured(self) -> bool:
        """Whether an API key has been set."""
        return self._llm_config.is_configured

    @property
    def is_context_loaded(self) -> bool:
        """Whether analysis data has been ingested."""
        return self._is_context_loaded

    @property
    def provider(self) -> str:
        return self._llm_config.provider.value

    @property
    def model(self) -> str:
        return self._llm_config.model

    def configure(
        self,
        api_key: Optional[str] = None,
        provider: Optional[str] = None,
        model: Optional[str] = None,
        base_url: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Update LLM configuration at runtime.

        Returns:
            The new configuration state.
        """
        if api_key is not None:
            self._llm_config.api_key = api_key
        if provider is not None:
            self._llm_config.provider = LLMProvider(provider)
            # Re-apply provider defaults for model/base_url if not overridden
            from .llm_service import _PROVIDER_DEFAULTS
            defaults = _PROVIDER_DEFAULTS.get(self._llm_config.provider, {})
            if not model and not self._llm_config.model:
                self._llm_config.model = defaults.get("model", "")
            if not base_url and not self._llm_config.base_url:
                self._llm_config.base_url = defaults.get("base_url", "")
        if model is not None:
            self._llm_config.model = model
        if base_url is not None:
            self._llm_config.base_url = base_url
        if temperature is not None:
            self._llm_config.temperature = temperature
        if max_tokens is not None:
            self._llm_config.max_tokens = max_tokens

        # Recreate LLM client with new config
        self._llm = LLMService(self._llm_config)

        return self.get_config_status()

    def get_config_status(self) -> Dict[str, Any]:
        """Return the current configuration (key is masked)."""
        key = self._llm_config.api_key
        masked_key = f"{key[:8]}...{key[-4:]}" if len(key) > 12 else "***"
        return {
            "is_configured": self.is_configured,
            "is_context_loaded": self._is_context_loaded,
            "provider": self._llm_config.provider.value,
            "model": self._llm_config.model,
            "base_url": self._llm_config.base_url,
            "temperature": self._llm_config.temperature,
            "max_tokens": self._llm_config.max_tokens,
            "api_key": masked_key if self.is_configured else None,
            "active_sessions": self._conversation_mgr.active_session_count,
        }

    # ── data context loading ─────────────────────────────────────────────

    def load_analysis_context(
        self,
        inventory_results: Optional[Dict] = None,
        bcg_results: Optional[Dict] = None,
        datasets: Optional[Dict] = None,
        custom_sections: Optional[Dict[str, str]] = None,
    ) -> str:
        """
        Build the data-aware system prompt from analysis results.

        Call this after running an analysis so the chatbot "knows"
        the current numbers.

        Returns:
            The assembled system prompt string.
        """
        self._context_builder.clear()

        if datasets:
            self._context_builder.ingest_raw_data_summary(datasets)
        if inventory_results:
            self._context_builder.ingest_inventory_results(inventory_results)
        if bcg_results:
            self._context_builder.ingest_bcg_results(bcg_results)
        if custom_sections:
            for key, text in custom_sections.items():
                self._context_builder.ingest_custom_section(key, text)

        system_prompt = self._context_builder.build_system_prompt()
        self._conversation_mgr.system_prompt = system_prompt

        # Propagate the new prompt to all existing sessions so
        # sessions created before data finished loading get the
        # full context on their next message.
        for sid in [s["session_id"] for s in self._conversation_mgr.list_sessions()]:
            self._conversation_mgr.update_system_prompt_for_session(
                sid, system_prompt
            )

        self._is_context_loaded = True

        logger.info(
            "Context loaded: %d chars, sections=%s",
            len(system_prompt),
            self._context_builder.get_section_keys(),
        )
        return system_prompt

    # ── session management ───────────────────────────────────────────────

    def create_session(
        self,
        session_id: Optional[str] = None,
        metadata: Optional[Dict] = None,
    ) -> str:
        """Create a new chat session. Returns the session ID."""
        return self._conversation_mgr.create_session(session_id, metadata)

    def get_or_create_session(self, session_id: Optional[str] = None) -> str:
        """Get existing session or create a new one."""
        if session_id and self._conversation_mgr.get_session(session_id):
            return session_id
        return self.create_session(session_id)

    def delete_session(self, session_id: str) -> bool:
        return self._conversation_mgr.delete_session(session_id)

    def reset_session(self, session_id: str) -> None:
        self._conversation_mgr.reset_session(session_id)

    def get_history(
        self, session_id: str, include_system: bool = False
    ) -> List[Dict[str, Any]]:
        return self._conversation_mgr.get_history(
            session_id, include_system=include_system
        )

    def list_sessions(self) -> List[Dict[str, Any]]:
        return self._conversation_mgr.list_sessions()

    # ── messaging ────────────────────────────────────────────────────────

    async def send_message(
        self,
        message: str,
        session_id: Optional[str] = None,
        *,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Send a user message and get the assistant's reply.

        If no ``session_id`` is given, a new session is created.

        Returns::

            {
                "session_id": str,
                "reply": str,
                "usage": dict,
                "model": str,
                "provider": str,
            }
        """
        if not self.is_configured:
            raise LLMError(
                "LLM not configured. Set the API key via "
                "ChatService.configure(api_key=...) or env var LLM_API_KEY."
            )

        # Ensure session
        sid = self.get_or_create_session(session_id)

        # Record user message
        self._conversation_mgr.add_user_message(sid, message)

        # Build message payload
        llm_messages = self._conversation_mgr.get_llm_messages(sid)

        # Call LLM
        result = await self._llm.chat(
            llm_messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )

        # Record assistant reply
        self._conversation_mgr.add_assistant_message(
            sid,
            result["content"],
            metadata={
                "model": result.get("model"),
                "usage": result.get("usage"),
            },
        )

        return {
            "session_id": sid,
            "reply": result["content"],
            "usage": result.get("usage", {}),
            "model": result.get("model", self._llm_config.model),
            "provider": result.get("provider", self._llm_config.provider.value),
        }

    async def stream_message(
        self,
        message: str,
        session_id: Optional[str] = None,
        *,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> AsyncIterator[str]:
        """
        Stream a response token-by-token.

        Yields partial text chunks. The full message is appended to
        history once the stream completes.
        """
        if not self.is_configured:
            raise LLMError("LLM not configured.")

        sid = self.get_or_create_session(session_id)
        self._conversation_mgr.add_user_message(sid, message)
        llm_messages = self._conversation_mgr.get_llm_messages(sid)

        full_response: list[str] = []
        async for token in self._llm.stream(
            llm_messages,
            temperature=temperature,
            max_tokens=max_tokens,
        ):
            full_response.append(token)
            yield token

        # Persist full response in history
        self._conversation_mgr.add_assistant_message(
            sid, "".join(full_response)
        )

    # ── health & stats ───────────────────────────────────────────────────

    async def health_check(self) -> Dict[str, Any]:
        """Check if the LLM connection is working."""
        return await self._llm.health()

    def get_usage_stats(self) -> Dict[str, Any]:
        """Cumulative token usage for the current process lifetime."""
        return self._llm.get_usage_stats()

    async def close(self) -> None:
        """Cleanup resources."""
        await self._llm.close()

    # ── cleanup ──────────────────────────────────────────────────────────

    def cleanup_expired_sessions(self) -> int:
        return self._conversation_mgr.cleanup_expired()

    def __repr__(self) -> str:
        return (
            f"ChatService(provider={self.provider!r}, "
            f"model={self.model!r}, "
            f"configured={self.is_configured})"
        )
