"""Synthesiser agent: produce the final grounded answer from accumulated summaries."""

from __future__ import annotations

from langchain_core.prompts import ChatPromptTemplate

from agents.state import ResearchState
from config import get_llm

_SYSTEM_PROMPT = """You are the SYNTHESISER agent on a research team.

You receive the user's original research query and a set of per-batch
summaries produced by the Summariser. Your job is to produce the final,
user-facing answer.

Requirements:
- Directly answer the user's query in a clear, well-structured response.
- Ground every factual claim in the supplied summaries; do not invent facts.
- Where helpful, organise the answer with short headings or bullets.
- End with a short "Sources" section listing the citation indices that were
  referenced (e.g. `[1], [3]`). If no citations appear in the summaries,
  omit the Sources section.
- If the summaries are insufficient to answer the query, say so honestly and
  explain what is missing."""

_USER_TEMPLATE = """Research query: {query}

Summaries:
{summaries_block}

Write the final answer now."""


def _format_summaries(summaries: list[str]) -> str:
    if not summaries:
        return "(no summaries provided)"
    blocks = [f"--- Summary batch {idx} ---\n{text}" for idx, text in enumerate(summaries, start=1)]
    return "\n\n".join(blocks)


def synthesiser_node(state: ResearchState) -> dict:
    """Generate the final ``state["final_answer"]`` from accumulated summaries.

    Uses a streaming-capable LLM so that the CLI can render tokens as they
    arrive (see :mod:`main`).
    """
    summaries = list(state.get("summaries", []) or [])

    prompt = ChatPromptTemplate.from_messages(
        [("system", _SYSTEM_PROMPT), ("user", _USER_TEMPLATE)]
    )
    chain = prompt | get_llm(streaming=True)

    response = chain.invoke(
        {
            "query": state["query"],
            "summaries_block": _format_summaries(summaries),
        }
    )
    text = response.content if isinstance(response.content, str) else str(response.content)
    return {"final_answer": text.strip()}
