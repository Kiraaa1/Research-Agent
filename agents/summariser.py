"""Summariser agent: condense raw search results into compact bullet notes."""

from __future__ import annotations

from langchain_core.prompts import ChatPromptTemplate

from agents.state import ResearchState
from config import get_llm
from tools.search import format_results_for_prompt

_SYSTEM_PROMPT = """You are the SUMMARISER agent on a research team.

Given the user's original research query and a batch of raw web-search hits,
produce a concise, factual digest:

- Extract only information relevant to the query.
- Preserve specific facts, numbers, names, and dates when present.
- Use compact bullet points (no preamble, no closing remarks).
- Cite the source index as `[1]`, `[2]`, etc., matching the input ordering.
- If a hit is irrelevant or empty, omit it silently.
- Do NOT speculate or add information that isn't in the inputs."""

_USER_TEMPLATE = """Research query: {query}

Search results:
{results_block}

Write the summary now."""


def summariser_node(state: ResearchState) -> dict:
    """Summarise the not-yet-summarised portion of ``state["search_results"]``.

    To avoid re-summarising the same hits when the supervisor loops, we only
    feed in results that arrived since the last summarisation pass.
    """
    all_results = list(state.get("search_results", []) or [])
    summaries_so_far = list(state.get("summaries", []) or [])

    if not all_results:
        return {"summaries": ["(no search results were available to summarise)"]}

    # Heuristic: each summarisation pass consumes whatever results exist now.
    # Pass `len(summaries_so_far)` as a lower-bound offset to avoid duplicates
    # when the supervisor re-invokes the searcher between summarisations.
    offset = min(len(summaries_so_far), max(len(all_results) - 1, 0))
    pending = all_results[offset:] if offset else all_results

    prompt = ChatPromptTemplate.from_messages(
        [("system", _SYSTEM_PROMPT), ("user", _USER_TEMPLATE)]
    )
    chain = prompt | get_llm(streaming=False)

    response = chain.invoke(
        {
            "query": state["query"],
            "results_block": format_results_for_prompt(pending),
        }
    )

    text = response.content if isinstance(response.content, str) else str(response.content)
    return {"summaries": [text.strip()]}
