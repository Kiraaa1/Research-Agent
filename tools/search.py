"""Web-search tool factory and result normalisation helpers.

We support Tavily (recommended) and DuckDuckGo. Both providers return slightly
different payload shapes, so :func:`parse_search_results` exposes a single
normalised format used by the rest of the graph.
"""

from __future__ import annotations

import ast
import json
import re
from typing import Any, Iterable

from langchain_core.tools import BaseTool

from config import max_search_results, search_provider


def build_search_tool() -> BaseTool:
    """Return a LangChain tool for web search based on the active provider.

    The tool is bindable to an LLM (so the Searcher agent can emit tool calls)
    and is also directly executable via ``langgraph.prebuilt.ToolNode``.
    """
    provider = search_provider()
    if provider == "tavily":
        from langchain_community.tools.tavily_search import TavilySearchResults

        return TavilySearchResults(
            max_results=max_search_results(),
            name="web_search",
            description=(
                "Search the web for up-to-date information. "
                "Input should be a focused natural-language query."
            ),
        )

    from langchain_community.tools import DuckDuckGoSearchResults

    kwargs: dict[str, Any] = {
        "num_results": max_search_results(),
        "name": "web_search",
        "description": (
            "Search the web for up-to-date information. "
            "Input should be a focused natural-language query."
        ),
    }
    # output_format was added in langchain-community 0.3.x; older releases
    # raise on unknown fields, so probe before passing it.
    try:
        return DuckDuckGoSearchResults(**kwargs, output_format="list")
    except (TypeError, ValueError):
        return DuckDuckGoSearchResults(**kwargs)


def _coerce_to_list(raw: Any) -> list[dict[str, Any]]:
    """Best-effort conversion of a tool payload into a list of result dicts."""
    if raw is None:
        return []
    if isinstance(raw, list):
        return [item for item in raw if isinstance(item, dict)]
    if isinstance(raw, dict):
        return [raw]
    if isinstance(raw, str):
        text = raw.strip()
        if not text:
            return []
        for parser in (json.loads, ast.literal_eval):
            try:
                parsed = parser(text)
            except (ValueError, SyntaxError):
                continue
            return _coerce_to_list(parsed)
        # DuckDuckGo's plain-text format: "snippet: ... , title: ... , link: ..."
        return _parse_duckduckgo_text(text)
    return []


_DDG_PATTERN = re.compile(
    r"snippet:\s*(?P<content>.*?),\s*title:\s*(?P<title>.*?),\s*link:\s*(?P<url>\S+)",
    re.DOTALL,
)


def _parse_duckduckgo_text(text: str) -> list[dict[str, Any]]:
    matches = _DDG_PATTERN.findall(text)
    return [
        {"content": content.strip(), "title": title.strip(), "url": url.strip()}
        for content, title, url in matches
    ]


def parse_search_results(raw: Any) -> list[dict[str, str]]:
    """Normalise a tool payload into ``[{title, url, content}]`` records.

    Both Tavily and DuckDuckGo loosely use these keys, but values may be
    missing or stored under aliases (``link``/``href``/``snippet``). We
    canonicalise them here so downstream agents have a stable contract.
    """
    items = _coerce_to_list(raw)
    normalised: list[dict[str, str]] = []
    for item in items:
        url = str(item.get("url") or item.get("link") or item.get("href") or "").strip()
        title = str(item.get("title") or item.get("name") or "").strip()
        content = str(
            item.get("content")
            or item.get("snippet")
            or item.get("body")
            or item.get("description")
            or ""
        ).strip()
        if not (url or title or content):
            continue
        normalised.append({"title": title, "url": url, "content": content})
    return normalised


def format_results_for_prompt(results: Iterable[dict[str, str]]) -> str:
    """Render normalised results into a numbered block suitable for an LLM prompt."""
    lines: list[str] = []
    for idx, result in enumerate(results, start=1):
        title = result.get("title") or "(untitled)"
        url = result.get("url") or "(no url)"
        content = result.get("content") or "(no content)"
        lines.append(f"[{idx}] {title}\n    URL: {url}\n    {content}")
    return "\n\n".join(lines) if lines else "(no results)"
