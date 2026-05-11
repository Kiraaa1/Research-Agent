"""Supervisor agent: routes the workflow to the next worker (or finishes).

The supervisor is an LLM that produces a structured ``SupervisorDecision``
choosing among ``searcher``, ``summariser``, ``synthesiser`` and ``FINISH``.
Hard guardrails (iteration cap and final-answer presence) override the LLM
to guarantee termination.
"""

from __future__ import annotations

from typing import cast

from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

from agents.state import (
    SUPERVISOR_DESTINATIONS,
    ResearchState,
    SupervisorDestination,
)
from config import get_llm, supervisor_max_iterations

_SYSTEM_PROMPT = """You are the SUPERVISOR of a research assistant team.

You coordinate three worker agents. On every turn you must decide which worker
should act next, or decide that the workflow is FINISHED.

Workers:
- "searcher": Performs web searches and adds hits to `search_results`.
  Use when `search_results` is empty, or when existing results are
  insufficient to cover the user's query.
- "summariser": Condenses raw `search_results` into compact `summaries`.
  Use after the searcher has returned results and before synthesis.
- "synthesiser": Combines `summaries` into a single grounded `final_answer`.
  Use once you have at least one summary and feel ready to answer.
- "FINISH": Stop the workflow. Only use when `final_answer` has already been
  written by the Synthesiser.

Rules:
1. Never invoke the synthesiser before at least one summary exists.
2. Never invoke the summariser when `search_results` is empty.
3. Prefer to FINISH as soon as a `final_answer` is present.
4. If the iteration count is near the cap, force progress toward FINISH.

Respond ONLY with the structured decision."""

_USER_TEMPLATE = """Research query: {query}

State snapshot:
- search_results collected: {n_results}
- summaries produced: {n_summaries}
- final_answer present: {has_final}
- iteration: {iteration} / {max_iterations}

Choose the next agent."""


class SupervisorDecision(BaseModel):
    """Structured output emitted by the supervisor LLM."""

    next_agent: SupervisorDestination = Field(
        description=(
            "Which worker to invoke next. One of: searcher, summariser, "
            "synthesiser, FINISH."
        )
    )
    reasoning: str = Field(
        description="One concise sentence explaining the routing decision.",
        max_length=400,
    )


def _build_chain():
    llm = get_llm(streaming=False)
    structured = llm.with_structured_output(SupervisorDecision)
    prompt = ChatPromptTemplate.from_messages(
        [("system", _SYSTEM_PROMPT), ("user", _USER_TEMPLATE)]
    )
    return prompt | structured


def supervisor_node(state: ResearchState) -> dict:
    """Decide which worker (if any) should run next.

    The function applies deterministic guardrails first and only falls back
    to the LLM when the routing decision is non-trivial.
    """
    iteration = state.get("iteration", 0) + 1
    max_iterations = supervisor_max_iterations()
    n_results = len(state.get("search_results", []) or [])
    n_summaries = len(state.get("summaries", []) or [])
    has_final = bool(state.get("final_answer"))

    decision: SupervisorDestination
    note: str

    if has_final:
        decision, note = "FINISH", "Final answer present; finishing."
    elif iteration >= max_iterations:
        if n_summaries:
            decision, note = "synthesiser", "Iteration cap reached; forcing synthesis."
        elif n_results:
            decision, note = "summariser", "Iteration cap reached; summarising what we have."
        else:
            decision, note = "FINISH", "Iteration cap reached with no data; aborting."
    else:
        chain = _build_chain()
        raw = chain.invoke(
            {
                "query": state["query"],
                "n_results": n_results,
                "n_summaries": n_summaries,
                "has_final": has_final,
                "iteration": iteration,
                "max_iterations": max_iterations,
            }
        )
        candidate = cast(SupervisorDecision, raw)
        decision = candidate.next_agent
        note = candidate.reasoning

        # Enforce invariants regardless of what the LLM chose.
        if decision == "summariser" and n_results == 0:
            decision, note = "searcher", "Override: nothing to summarise yet; searching."
        elif decision == "synthesiser" and n_summaries == 0:
            if n_results:
                decision, note = "summariser", "Override: need summaries before synthesis."
            else:
                decision, note = "searcher", "Override: need results before synthesis."
        elif decision == "FINISH" and not has_final:
            if n_summaries:
                decision, note = "synthesiser", "Override: must synthesise before finishing."
            elif n_results:
                decision, note = "summariser", "Override: must summarise before finishing."
            else:
                decision, note = "searcher", "Override: must search before finishing."

    if decision not in SUPERVISOR_DESTINATIONS:
        decision = "FINISH"
        note = f"Override: invalid destination, finishing. ({note})"

    return {
        "next_agent": decision,
        "iteration": iteration,
        "supervisor_notes": [f"[step {iteration}] -> {decision}: {note}"],
    }


def route_from_supervisor(state: ResearchState) -> SupervisorDestination:
    """Conditional-edge function that reads the supervisor's routing decision."""
    return state.get("next_agent", "FINISH")
