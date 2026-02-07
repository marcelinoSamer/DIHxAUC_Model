"""
File: llm_service.py
Description: Provider-agnostic LLM service for the FlavorFlow chatbot.
Dependencies: httpx (async HTTP client)
Author: FlavorFlow Team

Supports multiple free-tier LLM providers:
  - Groq   (default) — free, fast, OpenAI-compatible
  - Google Gemini     — generous free tier
  - Any OpenAI-compatible endpoint (OpenRouter, Together, etc.)

Usage:
    >>> from src.services.llm_service import LLMService, LLMConfig
    >>> config = LLMConfig(api_key="gsk_...")
    >>> llm = LLMService(config)
    >>> reply = await llm.chat([{"role": "user", "content": "Hello"}])
"""

from __future__ import annotations

import httpx
import json
import time
import logging
import os
from enum import Enum
from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Dict, List, Optional
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

logger = logging.getLogger("flavorflow.llm")


# ─── Provider definitions ────────────────────────────────────────────────────

class LLMProvider(str, Enum):
    """Supported LLM providers."""
    GROQ = "groq"
    GEMINI = "gemini"
    OPENAI_COMPATIBLE = "openai_compatible"


# Provider-specific defaults
_PROVIDER_DEFAULTS: Dict[str, Dict[str, Any]] = {
    LLMProvider.GROQ: {
        "base_url": "https://api.groq.com/openai/v1",
        "model": "llama-3.3-70b-versatile",
        "max_tokens": 2048,
        "supports_streaming": True,
    },
    LLMProvider.GEMINI: {
        "base_url": "https://generativelanguage.googleapis.com/v1beta",
        "model": "gemini-2.0-flash",
        "max_tokens": 2048,
        "supports_streaming": True,
    },
    LLMProvider.OPENAI_COMPATIBLE: {
        "base_url": "https://api.openai.com/v1",
        "model": "gpt-3.5-turbo",
        "max_tokens": 2048,
        "supports_streaming": True,
    },
}


# ─── Configuration ───────────────────────────────────────────────────────────

@dataclass
class LLMConfig:
    """
    Configuration for the LLM service.

    At minimum, provide an ``api_key``.  Everything else has sensible
    defaults (Groq + llama-3.3-70b-versatile).

    Attributes:
        api_key:      API key for the chosen provider.
        provider:     One of ``groq``, ``gemini``, ``openai_compatible``.
        model:        Model identifier (provider-specific).
        base_url:     Override the provider's default base URL.
        max_tokens:   Maximum tokens to generate in a single response.
        temperature:  Sampling temperature (0 = deterministic, 1 = creative).
        timeout:      HTTP request timeout in seconds.
    """
    api_key: str = field(default_factory=lambda: os.getenv('LLM_API_KEY', ''))
    provider: LLMProvider = LLMProvider.GROQ
    model: str = ""
    base_url: str = ""
    max_tokens: int = 0
    temperature: float = 0.4
    timeout: float = 60.0

    def __post_init__(self) -> None:
        defaults = _PROVIDER_DEFAULTS.get(self.provider, {})
        if not self.model:
            self.model = defaults.get("model", "")
        if not self.base_url:
            self.base_url = defaults.get("base_url", "")
        if self.max_tokens == 0:
            self.max_tokens = defaults.get("max_tokens", 2048)

    @property
    def is_configured(self) -> bool:
        """Whether the minimal config (API key) is present."""
        return bool(self.api_key)


# ─── LLM Service ─────────────────────────────────────────────────────────────

class LLMService:
    """
    Async LLM client that talks to any supported provider.

    The public API is intentionally small:

    * ``chat()``   – send a list of messages, receive a complete reply.
    * ``stream()`` – same, but yield partial tokens as they arrive.
    * ``health()`` – lightweight check that the key / endpoint are valid.

    Internally, Groq and any OpenAI-compatible provider share the same
    code-path.  Gemini has its own formatter because the API shape differs.
    """

    def __init__(self, config: LLMConfig) -> None:
        self.config = config
        self._client: Optional[httpx.AsyncClient] = None
        self._usage_log: List[Dict[str, Any]] = []

    # ── lifecycle ─────────────────────────────────────────────────────────

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(timeout=self.config.timeout)
        return self._client

    async def close(self) -> None:
        """Close the underlying HTTP client."""
        if self._client and not self._client.is_closed:
            await self._client.aclose()

    # ── public API ────────────────────────────────────────────────────────

    async def chat(
        self,
        messages: List[Dict[str, str]],
        *,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Send *messages* and return the assistant's reply.

        Args:
            messages:    OpenAI-style list of ``{role, content}`` dicts.
            temperature: Override default temperature for this call.
            max_tokens:  Override default max_tokens for this call.

        Returns:
            ``{"content": str, "usage": dict, "model": str, "provider": str}``

        Raises:
            LLMError: On any provider / network error.
        """
        if not self.config.is_configured:
            raise LLMError("LLM API key not configured. Set LLM_API_KEY.")

        temp = temperature if temperature is not None else self.config.temperature
        tokens = max_tokens if max_tokens is not None else self.config.max_tokens

        if self.config.provider == LLMProvider.GEMINI:
            return await self._chat_gemini(messages, temp, tokens)
        return await self._chat_openai_compat(messages, temp, tokens)

    async def stream(
        self,
        messages: List[Dict[str, str]],
        *,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> AsyncIterator[str]:
        """
        Stream partial tokens from the LLM.

        Yields:
            Token strings as they arrive.
        """
        if not self.config.is_configured:
            raise LLMError("LLM API key not configured. Set LLM_API_KEY.")

        temp = temperature if temperature is not None else self.config.temperature
        tokens = max_tokens if max_tokens is not None else self.config.max_tokens

        if self.config.provider == LLMProvider.GEMINI:
            async for chunk in self._stream_gemini(messages, temp, tokens):
                yield chunk
        else:
            async for chunk in self._stream_openai_compat(messages, temp, tokens):
                yield chunk

    async def health(self) -> Dict[str, Any]:
        """
        Quick health check: confirms the key + endpoint work.

        Returns:
            ``{"ok": bool, "provider": str, "model": str, "error": str|None}``
        """
        try:
            result = await self.chat(
                [{"role": "user", "content": "Reply with OK"}],
                max_tokens=5,
            )
            return {
                "ok": True,
                "provider": self.config.provider.value,
                "model": self.config.model,
                "error": None,
            }
        except Exception as exc:
            return {
                "ok": False,
                "provider": self.config.provider.value,
                "model": self.config.model,
                "error": str(exc),
            }

    def get_usage_stats(self) -> Dict[str, Any]:
        """Return cumulative token usage across all calls."""
        total_prompt = sum(u.get("prompt_tokens", 0) for u in self._usage_log)
        total_completion = sum(u.get("completion_tokens", 0) for u in self._usage_log)
        return {
            "total_calls": len(self._usage_log),
            "total_prompt_tokens": total_prompt,
            "total_completion_tokens": total_completion,
            "total_tokens": total_prompt + total_completion,
        }

    # ── OpenAI-compatible path (Groq, OpenRouter, Together, …) ───────────

    async def _chat_openai_compat(
        self, messages: List[Dict], temp: float, tokens: int
    ) -> Dict[str, Any]:
        client = await self._get_client()
        url = f"{self.config.base_url.rstrip('/')}/chat/completions"

        payload = {
            "model": self.config.model,
            "messages": messages,
            "temperature": temp,
            "max_tokens": tokens,
        }
        headers = {
            "Authorization": f"Bearer {self.config.api_key}",
            "Content-Type": "application/json",
        }

        start = time.monotonic()
        resp = await client.post(url, json=payload, headers=headers)
        elapsed = time.monotonic() - start

        if resp.status_code != 200:
            raise LLMError(
                f"[{self.config.provider.value}] HTTP {resp.status_code}: "
                f"{resp.text[:500]}"
            )

        data = resp.json()
        content = data["choices"][0]["message"]["content"]
        usage = data.get("usage", {})
        self._usage_log.append({**usage, "elapsed_s": round(elapsed, 2)})

        return {
            "content": content,
            "usage": usage,
            "model": data.get("model", self.config.model),
            "provider": self.config.provider.value,
            "elapsed_s": round(elapsed, 2),
        }

    async def _stream_openai_compat(
        self, messages: List[Dict], temp: float, tokens: int
    ) -> AsyncIterator[str]:
        client = await self._get_client()
        url = f"{self.config.base_url.rstrip('/')}/chat/completions"

        payload = {
            "model": self.config.model,
            "messages": messages,
            "temperature": temp,
            "max_tokens": tokens,
            "stream": True,
        }
        headers = {
            "Authorization": f"Bearer {self.config.api_key}",
            "Content-Type": "application/json",
        }

        async with client.stream("POST", url, json=payload, headers=headers) as resp:
            if resp.status_code != 200:
                body = await resp.aread()
                raise LLMError(
                    f"[{self.config.provider.value}] HTTP {resp.status_code}: "
                    f"{body.decode()[:500]}"
                )
            async for line in resp.aiter_lines():
                if not line.startswith("data: "):
                    continue
                raw = line[6:]
                if raw.strip() == "[DONE]":
                    break
                try:
                    chunk = json.loads(raw)
                    delta = chunk["choices"][0].get("delta", {})
                    token = delta.get("content")
                    if token:
                        yield token
                except (json.JSONDecodeError, KeyError, IndexError):
                    continue

    # ── Google Gemini path ────────────────────────────────────────────────

    def _gemini_format_messages(
        self, messages: List[Dict[str, str]]
    ) -> tuple[Optional[str], list[dict]]:
        """Convert OpenAI-style messages to Gemini format."""
        system_text: Optional[str] = None
        contents: list[dict] = []

        for msg in messages:
            role = msg["role"]
            text = msg["content"]
            if role == "system":
                system_text = text
            elif role == "assistant":
                contents.append({"role": "model", "parts": [{"text": text}]})
            else:
                contents.append({"role": "user", "parts": [{"text": text}]})

        return system_text, contents

    async def _chat_gemini(
        self, messages: List[Dict], temp: float, tokens: int
    ) -> Dict[str, Any]:
        client = await self._get_client()
        system_text, contents = self._gemini_format_messages(messages)

        url = (
            f"{self.config.base_url.rstrip('/')}"
            f"/models/{self.config.model}:generateContent"
            f"?key={self.config.api_key}"
        )

        payload: Dict[str, Any] = {
            "contents": contents,
            "generationConfig": {
                "temperature": temp,
                "maxOutputTokens": tokens,
            },
        }
        if system_text:
            payload["systemInstruction"] = {
                "parts": [{"text": system_text}]
            }

        start = time.monotonic()
        resp = await client.post(url, json=payload)
        elapsed = time.monotonic() - start

        if resp.status_code != 200:
            raise LLMError(
                f"[gemini] HTTP {resp.status_code}: {resp.text[:500]}"
            )

        data = resp.json()
        content = (
            data["candidates"][0]["content"]["parts"][0]["text"]
        )
        usage = data.get("usageMetadata", {})
        mapped_usage = {
            "prompt_tokens": usage.get("promptTokenCount", 0),
            "completion_tokens": usage.get("candidatesTokenCount", 0),
            "total_tokens": usage.get("totalTokenCount", 0),
        }
        self._usage_log.append({**mapped_usage, "elapsed_s": round(elapsed, 2)})

        return {
            "content": content,
            "usage": mapped_usage,
            "model": self.config.model,
            "provider": "gemini",
            "elapsed_s": round(elapsed, 2),
        }

    async def _stream_gemini(
        self, messages: List[Dict], temp: float, tokens: int
    ) -> AsyncIterator[str]:
        client = await self._get_client()
        system_text, contents = self._gemini_format_messages(messages)

        url = (
            f"{self.config.base_url.rstrip('/')}"
            f"/models/{self.config.model}:streamGenerateContent"
            f"?alt=sse&key={self.config.api_key}"
        )

        payload: Dict[str, Any] = {
            "contents": contents,
            "generationConfig": {
                "temperature": temp,
                "maxOutputTokens": tokens,
            },
        }
        if system_text:
            payload["systemInstruction"] = {
                "parts": [{"text": system_text}]
            }

        async with client.stream("POST", url, json=payload) as resp:
            if resp.status_code != 200:
                body = await resp.aread()
                raise LLMError(
                    f"[gemini] HTTP {resp.status_code}: {body.decode()[:500]}"
                )
            async for line in resp.aiter_lines():
                if not line.startswith("data: "):
                    continue
                raw = line[6:]
                try:
                    chunk = json.loads(raw)
                    parts = (
                        chunk.get("candidates", [{}])[0]
                        .get("content", {})
                        .get("parts", [])
                    )
                    for part in parts:
                        text = part.get("text")
                        if text:
                            yield text
                except (json.JSONDecodeError, KeyError, IndexError):
                    continue


# ─── Errors ──────────────────────────────────────────────────────────────────

class LLMError(Exception):
    """Raised when an LLM call fails."""
