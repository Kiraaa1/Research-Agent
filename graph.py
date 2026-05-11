"""Assemble the multi-agent research ``StateGraph``.

Topology
========

    START -> supervisor
    supervisor --(searcher)----> searcher -> search_tools -> collect_results -> supervisor
    supervisor --(summariser)--> summariser ---------------------------------> supervisor
    supervisor --(synthesiser)-> synthesiser --------------------------------> supervisor
    supervisor --(FINISH)------> END

The supervisor decides routing via a structured-output LLM; the
:func:`agents.supervisor.route_from_supervisor` conditional-edge function
reads the decision from state.
"""

from __future__ import annotations

from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.prebuilt import ToolNode

from agents import (
    ResearchState,
    collect_search_results,
    route_after_searcher,
    route_from_supervisor,
    searcher_node,
    summariser_node,
    supervisor_node,
    synthesiser_node,
)
from tools.search import build_search_tool


def build_graph() -> CompiledStateGraph:
    """Build and compile the multi-agent research graph.

    Returns:
        A compiled LangGraph runnable. Invoke it with an initial state
        containing at least ``{"query": "..."}``.
    """
    search_tool = build_search_tool()
    search_tools_node = ToolNode([search_tool])

    workflow: StateGraph = StateGraph(ResearchState)

    workflow.add_node("supervisor", supervisor_node)
    workflow.add_node("searcher", searcher_node)
    workflow.add_node("search_tools", search_tools_node)
    workflow.add_node("collect_results", collect_search_results)
    workflow.add_node("summariser", summariser_node)
    workflow.add_node("synthesiser", synthesiser_node)

    workflow.add_edge(START, "supervisor")

    workflow.add_conditional_edges(
        "supervisor",
        route_from_supervisor,
        {
            "searcher": "searcher",
            "summariser": "summariser",
            "synthesiser": "synthesiser",
            "FINISH": END,
        },
    )

    # Searcher emits tool calls -> ToolNode executes -> collector normalises.
    # If the LLM declines to call the tool, skip ToolNode and bounce back to
    # the supervisor instead of erroring out.
    workflow.add_conditional_edges(
        "searcher",
        route_after_searcher,
        {
            "search_tools": "search_tools",
            "supervisor": "supervisor",
        },
    )
    workflow.add_edge("search_tools", "collect_results")
    workflow.add_edge("collect_results", "supervisor")

    workflow.add_edge("summariser", "supervisor")
    workflow.add_edge("synthesiser", "supervisor")

    return workflow.compile()


__all__ = ["build_graph"]
