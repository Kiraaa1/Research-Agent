"""Web Searcher agent.

The searcher is an LLM with the web-search tool bound to it. It emits an
``AIMessage`` containing one or more tool calls; the graph's ``ToolNode``
then executes those calls and appends ``ToolMessage`` results. Finally, the
:func:`collect_search_results` node parses those tool messages into the
structured ``state["search_results"]`` list.
"""

from __future__ import annotations

from typing import cast

from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)

from agents.state import ResearchState, SearchResult
from config import get_llm
from tools.search import build_search_tool, parse_search_results


def route_after_searcher(state: ResearchState) -> str:
    """Conditional edge: only invoke ``ToolNode`` if the LLM emitted tool calls.

    Returns either ``"search_tools"`` (run tools) or ``"supervisor"`` (skip
    tool execution and let the supervisor re-route).
    """
    messages = state.get("messages", []) or []
    if not messages:
        return "supervisor"
    last = messages[-1]
    if isinstance(last, AIMessage) and getattr(last, "tool_calls", None):
        return "search_tools"
    return "supervisor"

_SYSTEM_PROMPT = """You are the WEB SEARCHER agent on a research team.

Given the user's research query (and any prior context), call the `web_search`
tool with one or two focused, well-formed natural-language search queries.

Guidelines:
- Prefer specific, factual queries over broad ones.
- If the user's question spans multiple sub-topics, issue separate searches.
- Do NOT answer the user yourself; only emit tool calls.
- If you have already retrieved sufficient results, you may respond with
  plain text and no tool calls; an upstream supervisor will move on.

Never call any tool other than `web_search`."""


def searcher_node(state: ResearchState) -> dict:
    """Generate one or more ``web_search`` tool calls for the current query."""
    llm = get_llm(streaming=False)
    tool = build_search_tool()
    llm_with_tools = llm.bind_tools([tool])

    existing_results = len(state.get("search_results", []) or [])
    context_hint = (
        "No searches have been run yet."
        if existing_results == 0
        else (
            f"{existing_results} results have already been collected; "
            "only search again if a new angle is required."
        )
    )

    messages: list[BaseMessage] = [
        SystemMessage(content=_SYSTEM_PROMPT),
        HumanMessage(
            content=(
                f"Research query: {state['query']}\n\n"
                f"Context: {context_hint}\n\n"
                "Issue the appropriate `web_search` tool call(s)."
            )
        ),
    ]

    response = llm_with_tools.invoke(messages)
    return {"messages": [response]}


def _latest_tool_messages(messages: list[BaseMessage]) -> list[ToolMessage]:
    """Return the trailing run of ``ToolMessage`` instances (most recent batch)."""
    collected: list[ToolMessage] = []
    for message in reversed(messages):
        if isinstance(message, ToolMessage):
            collected.append(message)
        elif isinstance(message, AIMessage):
            break
        else:
            break
    collected.reverse()
    return collected


def collect_search_results(state: ResearchState) -> dict:
    """Parse the most recent ``ToolMessage`` batch into structured results.

    Runs immediately after the ``ToolNode`` so we can drop the raw tool output
    from the conversational scratchpad and instead keep a clean, typed list of
    hits in ``state["search_results"]``.
    """
    messages = list(state.get("messages", []) or [])
    tool_messages = _latest_tool_messages(messages)
    if not tool_messages:
        return {}

    new_results: list[SearchResult] = []
    for tool_message in tool_messages:
        parsed = parse_search_results(tool_message.content)
        for hit in parsed:
            new_results.append(cast(SearchResult, hit))

    return {"search_results": new_results}
