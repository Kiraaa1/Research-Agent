"""Runtime configuration for the multi-agent research assistant.

Centralises environment variable loading and the LLM factory so individual
agents do not duplicate provider-selection logic.
"""

from __future__ import annotations

import os
from functools import lru_cache
from typing import Literal

from dotenv import load_dotenv
from langchain_core.language_models import BaseChatModel

load_dotenv()

LLMProvider = Literal["openai", "anthropic"]
SearchProvider = Literal["tavily", "duckduckgo"]


def _provider() -> LLMProvider:
    raw = os.getenv("LLM_PROVIDER", "openai").strip().lower()
    if raw not in {"openai", "anthropic"}:
        raise ValueError(
            f"Unsupported LLM_PROVIDER {raw!r}. Use 'openai' or 'anthropic'."
        )
    return raw  # type: ignore[return-value]


def search_provider() -> SearchProvider:
    """Return the configured search provider, with auto-fallback to DuckDuckGo.

    Tavily is preferred for quality, but if ``TAVILY_API_KEY`` is missing we
    silently degrade to DuckDuckGo so the CLI still works out of the box.
    """
    raw = os.getenv("SEARCH_PROVIDER", "tavily").strip().lower()
    if raw == "tavily" and not os.getenv("TAVILY_API_KEY"):
        return "duckduckgo"
    if raw not in {"tavily", "duckduckgo"}:
        raise ValueError(
            f"Unsupported SEARCH_PROVIDER {raw!r}. Use 'tavily' or 'duckduckgo'."
        )
    return raw  # type: ignore[return-value]


def max_search_results() -> int:
    return int(os.getenv("MAX_SEARCH_RESULTS", "5"))


def supervisor_max_iterations() -> int:
    return int(os.getenv("SUPERVISOR_MAX_ITERATIONS", "8"))


@lru_cache(maxsize=4)
def get_llm(*, streaming: bool = False, temperature: float = 0.0) -> BaseChatModel:
    """Return a configured chat model for the active provider.

    Results are cached per ``(streaming, temperature)`` combination so we do not
    re-instantiate clients on every node invocation.
    """
    provider = _provider()
    if provider == "anthropic":
        from langchain_anthropic import ChatAnthropic

        model = os.getenv("ANTHROPIC_MODEL", "claude-3-5-sonnet-latest")
        return ChatAnthropic(
            model=model,
            temperature=temperature,
            streaming=streaming,
            max_tokens=2048,
        )

    from langchain_openai import ChatOpenAI

    model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    return ChatOpenAI(
        model=model,
        temperature=temperature,
        streaming=streaming,
    )
