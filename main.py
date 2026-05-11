"""CLI entrypoint for the multi-agent research assistant.

Usage:

    # interactive REPL
    python main.py

    # single shot
    python main.py "What are the latest trends in quantum error correction?"

The CLI streams the Synthesiser's tokens to the terminal as they are produced
and shows lightweight progress updates as each agent node runs.
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from typing import Any

from langchain_core.messages import AIMessageChunk
from rich.console import Console
from rich.panel import Panel

from agents.state import ResearchState
from config import supervisor_max_iterations
from graph import build_graph

console = Console()

_NODE_LABELS = {
    "supervisor": "Supervisor",
    "searcher": "Searcher",
    "search_tools": "Tool execution",
    "collect_results": "Collecting results",
    "summariser": "Summariser",
    "synthesiser": "Synthesiser",
}


async def run_query(query: str) -> str:
    """Run a single research query through the graph and stream output.

    Returns the final synthesised answer.
    """
    graph = build_graph()

    initial_state: ResearchState = {  # type: ignore[typeddict-item]
        "query": query,
        "messages": [],
        "search_results": [],
        "summaries": [],
        "supervisor_notes": [],
        "iteration": 0,
    }

    recursion_limit = max(25, supervisor_max_iterations() * 6)

    final_answer = ""
    streaming_synth = False

    async for stream_mode, payload in graph.astream(
        initial_state,
        config={"recursion_limit": recursion_limit},
        stream_mode=["updates", "messages"],
    ):
        if stream_mode == "updates":
            _handle_update(payload)
        elif stream_mode == "messages":
            chunk, metadata = payload
            if metadata.get("langgraph_node") != "synthesiser":
                continue
            if not isinstance(chunk, AIMessageChunk):
                continue
            piece = chunk.content if isinstance(chunk.content, str) else ""
            if not piece:
                continue
            if not streaming_synth:
                console.print()
                console.rule("[bold green]Final answer[/bold green]")
                streaming_synth = True
            console.print(piece, end="", soft_wrap=True, highlight=False)
            final_answer += piece

    if streaming_synth:
        console.print()

    return final_answer


def _handle_update(update: dict[str, Any]) -> None:
    """Print a single-line status update each time a node finishes."""
    for node_name, node_update in update.items():
        label = _NODE_LABELS.get(node_name, node_name)
        detail = _summarise_update(node_name, node_update)
        console.print(f"[dim]→[/dim] [bold cyan]{label}[/bold cyan] {detail}")


def _summarise_update(node_name: str, node_update: Any) -> str:
    if not isinstance(node_update, dict):
        return ""
    if node_name == "supervisor":
        notes = node_update.get("supervisor_notes") or []
        if notes:
            return f"[dim]{notes[-1]}[/dim]"
        next_agent = node_update.get("next_agent")
        return f"[dim]-> {next_agent}[/dim]" if next_agent else ""
    if node_name == "collect_results":
        n = len(node_update.get("search_results") or [])
        return f"[dim](+{n} results)[/dim]"
    if node_name == "summariser":
        n = len(node_update.get("summaries") or [])
        return f"[dim](+{n} summary batch)[/dim]"
    if node_name == "synthesiser":
        return "[dim](streaming...)[/dim]"
    return ""


def _print_banner() -> None:
    console.print(
        Panel.fit(
            "[bold]Multi-agent research assistant[/bold]\n"
            "[dim]LangGraph supervisor / searcher / summariser / synthesiser[/dim]",
            border_style="cyan",
        )
    )


async def _repl() -> None:
    _print_banner()
    while True:
        try:
            console.print()
            query = console.input("[bold]Query[/bold] ([dim]Ctrl-C to quit[/dim])› ").strip()
        except (EOFError, KeyboardInterrupt):
            console.print("\n[dim]Goodbye.[/dim]")
            return
        if not query:
            continue
        try:
            await run_query(query)
        except KeyboardInterrupt:
            console.print("\n[yellow]Cancelled.[/yellow]")
        except Exception as exc:  # noqa: BLE001 - top-level CLI guard
            console.print(f"[red]Error:[/red] {exc}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Multi-agent research assistant (LangGraph)."
    )
    parser.add_argument(
        "query",
        nargs="*",
        help="Research query. Omit for interactive REPL mode.",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    query = " ".join(args.query).strip()

    try:
        if query:
            _print_banner()
            asyncio.run(run_query(query))
        else:
            asyncio.run(_repl())
    except KeyboardInterrupt:
        console.print("\n[dim]Interrupted.[/dim]")
        return 130
    return 0


if __name__ == "__main__":
    sys.exit(main())
