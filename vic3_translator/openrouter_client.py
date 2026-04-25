"""OpenRouter API client — fetch and search available models."""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Optional
from urllib.request import Request, urlopen
from urllib.error import URLError


logger = logging.getLogger(__name__)

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
MODELS_ENDPOINT = f"{OPENROUTER_BASE_URL}/models"


def fetch_available_models(api_key: str) -> List[Dict[str, Any]]:
    """Fetch available models from OpenRouter API.

    Returns a list of model dicts, each containing at least:
        - ``id``        (str)  — e.g. "anthropic/claude-3.5-sonnet"
        - ``name``      (str)  — human-readable name
        - ``pricing``   (dict) — ``{prompt, completion}`` in USD per token
        - ``context_length`` (int)

    Raises ``RuntimeError`` on network or auth failure.
    """
    if not api_key:
        raise ValueError("OpenRouter API key is required")

    req = Request(
        MODELS_ENDPOINT,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="GET",
    )

    try:
        with urlopen(req, timeout=30) as resp:
            body = json.loads(resp.read().decode("utf-8"))
    except URLError as e:
        raise RuntimeError(f"Failed to fetch models from OpenRouter: {e}") from e
    except json.JSONDecodeError as e:
        raise RuntimeError(f"Invalid JSON response from OpenRouter: {e}") from e

    # Response body: {"data": [ ... ]}
    data: List[Dict[str, Any]] = body.get("data", [])
    logger.info("Fetched %d models from OpenRouter", len(data))
    return data


def search_models(
    query: str,
    models: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Filter ``models`` by ``query`` (case-insensitive match on id/name).

    Returns models sorted by id alphabetically.
    """
    q = query.strip().lower()
    if not q:
        return sorted(models, key=lambda m: m.get("id", ""))

    filtered = [
        m
        for m in models
        if q in m.get("id", "").lower() or q in m.get("name", "").lower()
    ]
    return sorted(filtered, key=lambda m: m.get("id", ""))


def format_model_label(model: Dict[str, Any]) -> str:
    """Format a model dict into a human-readable label string."""
    mid = model.get("id", "?")
    name = model.get("name", "")
    pricing = model.get("pricing", {})
    prompt_price = pricing.get("prompt", 0)
    completion_price = pricing.get("completion", 0)
    ctx = model.get("context_length", 0)
    price_str = f"${float(prompt_price):.6f}/in ${float(completion_price):.6f}/out"
    ctx_str = f"{int(ctx):,}" if ctx else "?"
    return f"{mid}  |  {ctx_str} ctx  |  {price_str}  |  {name}"