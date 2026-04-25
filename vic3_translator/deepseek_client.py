"""Thin wrapper around the DeepSeek (V4 Pro) chat-completions API.

DeepSeek's public endpoint is OpenAI-compatible, so we reuse the official
``openai`` Python SDK and only override ``base_url``.

Reference::

    from openai import OpenAI
    client = OpenAI(api_key=..., base_url="https://api.deepseek.com")
    client.chat.completions.create(model="deepseek-v4-pro", ...)

Model aliases (as of 2026):
    * ``deepseek-v4-pro``   — default, thinking mode capable, 1M context
    * ``deepseek-v4-flash`` — cheaper flash variant
    * ``deepseek-chat``     — legacy alias (non-thinking)
    * ``deepseek-reasoner`` — legacy alias (thinking)
"""

from __future__ import annotations

import json
import logging
import re
import time
from typing import Dict, List, Optional

try:
    from openai import OpenAI
    from openai import APIError, APIConnectionError, RateLimitError
except ImportError:  # pragma: no cover - surfaced at runtime
    OpenAI = None  # type: ignore
    APIError = Exception  # type: ignore
    APIConnectionError = Exception  # type: ignore
    RateLimitError = Exception  # type: ignore


from . import prompts


logger = logging.getLogger(__name__)


DEFAULT_BASE_URL = "https://api.deepseek.com"
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
DEFAULT_MODEL = "deepseek-v4-pro"


class DeepSeekError(RuntimeError):
    """Raised when the DeepSeek call repeatedly fails."""


class DeepSeekRateLimitError(DeepSeekError):
    """Raised when retries are exhausted due to rate limiting (HTTP 429)."""


class DeepSeekClient:
    """Translate batches of strings using DeepSeek."""

    def __init__(
        self,
        api_key: str,
        model: str = DEFAULT_MODEL,
        base_url: str = DEFAULT_BASE_URL,
        enable_thinking: bool = False,
        max_retries: int = 3,
        timeout: float = 180.0,
        rate_limit_policy: str = "backoff",  # "backoff" | "stop" | "ignore" | "retry"
    ):
        if OpenAI is None:
            raise RuntimeError(
                "The 'openai' package is not installed. "
                "Run: pip install -r requirements.txt"
            )
        if not api_key:
            raise ValueError("DeepSeek API key is required")

        self.model = model
        self.enable_thinking = enable_thinking
        self.max_retries = max_retries
        self.timeout = timeout
        self.rate_limit_policy = rate_limit_policy
        self.client = OpenAI(api_key=api_key, base_url=base_url, timeout=timeout)

        # Rate limiting state
        self._rate_limit_count = 0
        self._last_rate_limit = 0

    # ------------------------------------------------------------------ API
    def translate_batch(
        self,
        texts: List[str],
        source_lang: Dict[str, str],
        target_lang: Dict[str, str],
        mod_context: str = "",
    ) -> List[str]:
        """Translate ``texts`` (list of source strings) and return the
        list of translated strings. Length is guaranteed to match.

        Raises :class:`DeepSeekError` after all retries fail.
        """
        if not texts:
            return []

        system_prompt = prompts.build_system_prompt(source_lang, target_lang)
        user_prompt = prompts.build_user_prompt(
            texts, source_lang, target_lang, mod_context
        )

        last_err: Optional[Exception] = None
        retry_instruction: Optional[str] = None
        for attempt in range(1, self.max_retries + 1):
            try:
                raw = self._call(system_prompt, user_prompt, retry_instruction)
                parsed = self._parse_response(raw, expected=len(texts))
                if parsed is not None:
                    return parsed

                msg = f"DeepSeek response parse failed on attempt {attempt} (expected {len(texts)} items)"
                logger.warning(msg)
                last_err = DeepSeekError("Response parse failed")
                retry_instruction = (
                    f"IMPORTANT: Your previous response was invalid. "
                    f"The number of items must be EXACTLY {len(texts)}. "
                    f"Do not include any commentary, just return the JSON array."
                )
            except RateLimitError as e:
                logger.warning("Rate limited (attempt %d): %s", attempt, e)
                last_err = e
                time.sleep(min(30 * attempt, 90))
                continue
            except (APIError, APIConnectionError) as e:
                logger.warning("API error (attempt %d): %s", attempt, e)
                last_err = e
            except Exception as e:  # noqa: BLE001
                logger.exception("Unexpected error on attempt %d", attempt)
                last_err = e

            # exponential backoff (2, 4, 8, ...)
            if attempt < self.max_retries:
                delay = 2**attempt
                time.sleep(delay)

        if isinstance(last_err, RateLimitError):
            raise DeepSeekRateLimitError(
                f"DeepSeek rate-limited after {self.max_retries} attempts: {last_err}"
            )

        raise DeepSeekError(
            f"DeepSeek translation failed after {self.max_retries} attempts: {last_err}"
        )

    # -------------------------------------------------------------- helpers
    def _call(
        self, system_prompt: str, user_prompt: str, retry_msg: Optional[str] = None
    ) -> str:
        """Invoke the chat-completions endpoint and return the raw content."""
        extra_body: Dict[str, object] = {}
        # DeepSeek exposes an ``enable_thinking`` flag on the v4 family.
        extra_body["enable_thinking"] = bool(self.enable_thinking)

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        if retry_msg:
            messages.append({"role": "user", "content": retry_msg})

        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            max_tokens=8000,
            temperature=0.2,
            extra_body=extra_body,
        )
        content = response.choices[0].message.content or ""
        return content.strip()

    def _parse_response(self, content: str, expected: int) -> Optional[List[str]]:
        """Attempt to extract and parse a JSON array from the response content."""
        if not content:
            return None

        text = content.strip()
        try:
            # 1. Try to locate the outermost JSON array [...]
            # We use re.DOTALL to match across multiple lines.
            match = re.search(r'\[.*\]', text, re.DOTALL)
            if not match:
                # If no brackets found, maybe it's raw JSON without wrapping or wrapped in code fences
                if text.startswith("```"):
                    text = re.sub(r"^```[a-zA-Z]*\s*", "", text)
                    if text.endswith("```"):
                        text = text[:-3]
                    text = text.strip()
            else:
                text = match.group(0)

            # 2. Parse the JSON
            try:
                data = json.loads(text)
            except json.JSONDecodeError:
                # Lenient fix: strip trailing commas before closing brackets
                repaired = re.sub(r",\s*\]", "]", text)
                data = json.loads(repaired)

            if not isinstance(data, list):
                return None

            # 3. Validate length and coerce to strings
            items = [("" if item is None else str(item)) for item in data]
            if len(items) != expected:
                logger.warning("Parse failed: Expected %d items, got %d", expected, len(items))
                return None
            
            return items
        except Exception as e:
            logger.warning("JSON parse exception: %s\nRaw content (clipped): %s", e, content[:200])
            return None
