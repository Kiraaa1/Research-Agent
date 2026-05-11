"""Typed state schema shared across all nodes in the research graph.

The state is a :class:`TypedDict` so it integrates with LangGraph's reducer
system. List-valued fields use :func:`operator.add`/``add_messages`` reducers
so worker nodes can append rather than overwrite.
"""

from __future__ import annotations

from operator import add
from typing import Annotated, Literal, TypedDict

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages

SupervisorDestination = Literal[
    "searcher",
    "summariser",
    "synthesiser",
    "FINISH",
]

SUPERVISOR_DESTINATIONS: tuple[SupervisorDestination, ...] = (
    "searcher",
    "summariser",
    "synthesiser",
    "FINISH",
)


class SearchResult(TypedDict):
    """A single normalised web-search hit."""

    title: str
    url: str
    content: str


class ResearchState(TypedDict, total=False):
    """Shared state passed between supervisor and worker agents.

    Attributes:
        query: The original user research query.
        messages: Conversation buffer used for tool-calling within the
            searcher subgraph.
        search_results: Accumulated normalised search hits.
        summaries: Per-batch summaries produced by the Summariser.
        final_answer: The synthesised final answer (set once Synthesiser runs).
        next_agent: Routing decision emitted by the Supervisor.
        supervisor_notes: Free-form reasoning recorded by the Supervisor.
        iteration: Monotonically increasing supervisor step counter.
    """

    query: str
    messages: Annotated[list[BaseMessage], add_messages]
    search_results: Annotated[list[SearchResult], add]
    summaries: Annotated[list[str], add]
    final_answer: str
    next_agent: SupervisorDestination
    supervisor_notes: Annotated[list[str], add]
    iteration: int
