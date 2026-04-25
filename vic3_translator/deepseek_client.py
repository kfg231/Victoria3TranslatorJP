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
DEEPSEEK_DIRECT_MODEL_PREFIXES = ("deepseek-",)
DEEPSEEK_DIRECT_MODEL_ALIASES = {
    "deepseek-v4-pro",
    "deepseek-v4-flash",
    "deepseek-chat",
    "deepseek-reasoner",
}


class DeepSeekError(RuntimeError):
    """Raised when the DeepSeek call repeatedly fails."""


class DeepSeekRateLimitError(DeepSeekError):
    """Raised when retries are exhausted due to rate limiting (HTTP 429)."""


def is_openrouter_base_url(base_url: str) -> bool:
    return "openrouter" in str(base_url).lower()


def normalize_model_name(model: str, base_url: str) -> str:
    """Normalize model naming differences between providers."""
    normalized = (model or "").strip()
    if not normalized:
        return normalized

    if is_openrouter_base_url(base_url):
        return normalized

    if normalized.endswith(":thinking"):
        normalized = normalized[:-9]

    if "/" in normalized:
        candidate = normalized.rsplit("/", 1)[-1].strip()
        if candidate in DEEPSEEK_DIRECT_MODEL_ALIASES or candidate.startswith(
            DEEPSEEK_DIRECT_MODEL_PREFIXES
        ):
            normalized = candidate

    return normalized


class DeepSeekClient:
    """Translate batches of strings using DeepSeek."""

    def __init__(
        self,
        api_key: str,
        model: str = DEFAULT_MODEL,
        base_url: str = DEFAULT_BASE_URL,
        enable_thinking: bool = False,
        max_retries: int = 3,
        error_retries: int = 3,
        timeout: float = 180.0,
        rate_limit_policy: str = "backoff",  # "backoff" | "stop" | "ignore" | "retry"
        reasoning_effort: str = "none",  # "none" | "low" | "medium" | "high"
    ):
        if OpenAI is None:
            raise RuntimeError(
                "The 'openai' package is not installed. "
                "Run: pip install -r requirements.txt"
            )
        if not api_key:
            raise ValueError("DeepSeek API key is required")

        self.base_url = str(base_url)
        self.model = normalize_model_name(model, self.base_url)
        if self.model != model:
            logger.warning(
                "Normalized model for %s: %s -> %s",
                "OpenRouter" if is_openrouter_base_url(self.base_url) else "DeepSeek direct API",
                model,
                self.model,
            )
        self.enable_thinking = enable_thinking
        self.max_retries = max_retries
        self.error_retries = error_retries
        self.timeout = timeout
        self.rate_limit_policy = rate_limit_policy
        self.reasoning_effort = reasoning_effort
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
        max_retries: Optional[int] = None,
        error_retries: Optional[int] = None,
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
        # Internally treat these as "max attempts"
        max_429_attempts = max_retries if max_retries is not None else self.max_retries
        max_error_attempts = error_retries if error_retries is not None else self.error_retries
        
        attempt_429 = 0
        attempt_err = 0

        while True:
            try:
                raw = self._call(system_prompt, user_prompt, retry_instruction)
                parsed = self._parse_response(raw, expected=len(texts))
                if parsed is not None:
                    return parsed

                msg = f"DeepSeek response parse failed (expected {len(texts)} items)"
                logger.warning(msg)
                last_err = DeepSeekError("Response parse failed")
                retry_instruction = (
                    f"IMPORTANT: Your previous response was invalid. "
                    f"The number of items must be EXACTLY {len(texts)}. "
                    f"Do not include any commentary, just return the JSON array."
                )
                
                attempt_err += 1
                if attempt_err >= max_error_attempts:
                    raise DeepSeekError(f"DeepSeek translation failed after {max_error_attempts} error attempts: {last_err}")
                time.sleep(2**attempt_err)
                continue

            except RateLimitError as e:
                attempt_429 += 1
                logger.warning("Rate limited (attempt %d/%d): %s", attempt_429, max_429_attempts, e)
                last_err = e
                if attempt_429 >= max_429_attempts:
                    raise DeepSeekRateLimitError(f"DeepSeek rate-limited after {max_429_attempts} attempts: {last_err}")
                time.sleep(min(30 * attempt_429, 90))
                continue
            except (APIError, APIConnectionError) as e:
                err_msg = str(e)
                if self._maybe_recover_model_name(err_msg):
                    time.sleep(1)
                    continue

                non_retryable_reason = self._get_non_retryable_api_reason(e, err_msg)
                if non_retryable_reason:
                    raise DeepSeekError(f"{non_retryable_reason}: {e}") from e
                
                attempt_err += 1
                logger.warning("API error (attempt %d/%d): %s", attempt_err, max_error_attempts, e)
                last_err = e
                if attempt_err >= max_error_attempts:
                    raise DeepSeekError(f"DeepSeek translation failed after {max_error_attempts} error attempts: {last_err}")
                time.sleep(2**attempt_err)
                continue
            except Exception as e:  # noqa: BLE001
                attempt_err += 1
                logger.exception("Unexpected error (attempt %d/%d)", attempt_err, max_error_attempts)
                last_err = e
                if attempt_err >= max_error_attempts:
                    raise DeepSeekError(f"DeepSeek translation failed after {max_error_attempts} error attempts: {last_err}")
                time.sleep(2**attempt_err)
                continue

    # -------------------------------------------------------------- helpers
    def _call(
        self, system_prompt: str, user_prompt: str, retry_msg: Optional[str] = None
    ) -> str:
        """Invoke the chat-completions endpoint and return the raw content."""
        extra_body: Dict[str, object] = {}
        is_openrouter = is_openrouter_base_url(str(self.client.base_url))

        # DeepSeek direct API: use proprietary enable_thinking parameter
        if self.enable_thinking and not is_openrouter:
            extra_body["enable_thinking"] = True

        # OpenRouter: use the standardized reasoning parameter
        # See https://openrouter.ai/docs - reasoning.effort controls thinking intensity
        if is_openrouter and self.reasoning_effort and self.reasoning_effort != "none":
            extra_body["reasoning"] = {"effort": self.reasoning_effort}

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        if retry_msg:
            messages.append({"role": "user", "content": retry_msg})

        kwargs: Dict[str, object] = {
            "model": self.model,
            "messages": messages,
            "max_tokens": 12000,
            "temperature": 0.2,
        }
        if extra_body:
            kwargs["extra_body"] = extra_body

        response = self.client.chat.completions.create(**kwargs)
        content = response.choices[0].message.content or ""
        # Strip thinking blocks that some models wrap around the JSON output
        content = re.sub(r'<(think|思考|reasoning)>.*?</\1>', '', content, flags=re.DOTALL | re.IGNORECASE)
        return content.strip()

    def _maybe_recover_model_name(self, err_msg: str) -> bool:
        """Attempt a one-step recovery for common provider/model mismatches."""
        lower = err_msg.lower()
        missing_model = (
            "model not exist" in lower
            or "no endpoints found" in lower
            or "404" in lower
        )
        if not missing_model:
            return False

        if self.model.endswith(":thinking"):
            fallback_model = self.model[:-9]
            logger.warning(
                "Model %s does not support ':thinking'; falling back to %s",
                self.model,
                fallback_model,
            )
            self.model = fallback_model
            return True

        if not is_openrouter_base_url(self.base_url) and "/" in self.model:
            fallback_model = normalize_model_name(self.model, self.base_url)
            if fallback_model != self.model:
                logger.warning(
                    "Model %s looks like an OpenRouter id; retrying on DeepSeek direct API with %s",
                    self.model,
                    fallback_model,
                )
                self.model = fallback_model
                return True

        return False

    def _get_non_retryable_api_reason(
        self, error: Exception, err_msg: Optional[str] = None
    ) -> Optional[str]:
        """Return a reason when retrying is unlikely to help."""
        message = (err_msg or str(error)).lower()
        status_code = getattr(error, "status_code", None)
        response = getattr(error, "response", None)
        if status_code is None and response is not None:
            status_code = getattr(response, "status_code", None)

        if (
            "model not exist" in message
            or "no endpoints found" in message
            or "invalid_request_error" in message
        ):
            return "Model or request configuration error"

        if status_code is not None and 400 <= status_code < 500 and status_code not in (408, 409, 429):
            return f"Non-retryable API error ({status_code})"

        return None

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
                # Fallback: if it's a dict with a single list value, extract the list.
                if isinstance(data, dict):
                    lists = [v for v in data.values() if isinstance(v, list)]
                    if len(lists) == 1:
                        data = lists[0]
                    else:
                        return None
                else:
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
