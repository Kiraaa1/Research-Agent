# Multi-agent research assistant (LangGraph)

A small, production-quality research assistant built on
[LangGraph](https://github.com/langchain-ai/langgraph). A **Supervisor** agent
delegates each turn to one of three **Worker** agents — a **Web Searcher**, a
**Summariser**, and a **Synthesiser** — until a final answer is ready. The
final answer is streamed token-by-token to the terminal.

---

## Architecture

The system is modelled as a `StateGraph` with a typed `TypedDict` state shared
across all nodes. The Supervisor uses structured LLM output to decide which
worker should run next; deterministic guardrails ensure the graph always
terminates.

### Roles

| Agent           | Responsibility                                                                          |
| --------------- | --------------------------------------------------------------------------------------- |
| **Supervisor**  | Inspects the state and picks the next worker (`searcher` / `summariser` / `synthesiser` / `FINISH`). Enforces invariants (e.g. never synthesise before summaries exist) and an iteration cap. |
| **Searcher**    | LLM bound to a `web_search` tool. Emits `tool_calls`; the prebuilt `ToolNode` executes them and a small `collect_results` node normalises the payloads into a typed list. |
| **Summariser**  | Condenses freshly retrieved hits into compact, source-cited bullet notes.                |
| **Synthesiser** | Combines all summary batches into the final, grounded answer (streamed to the CLI).      |

### Graph topology

```
                        ┌─────────────┐
                        │    START    │
                        └──────┬──────┘
                               ▼
                        ┌─────────────┐
                ┌──────▶│ supervisor  │◀────────────────────┐
                │       └──────┬──────┘                      │
                │              │ conditional routing         │
                │   ┌──────────┼──────────────┐              │
                │   ▼          ▼              ▼              │
                │ ┌──────┐ ┌──────────┐ ┌─────────────┐      │
                │ │search│ │summariser│ │ synthesiser │      │
                │ │  er  │ └────┬─────┘ └──────┬──────┘      │
                │ └───┬──┘      │              │             │ 
                │     ▼         │              │             │ 
                │ ┌────────┐    │              │             │
                │ │search_ │    │              │             │
                │ │ tools  │    │              │             │
                │ │(ToolN.)│    │              │             │
                │ └───┬────┘    │              │             │
                │     ▼         │              │             │
                │ ┌────────┐    │              │             │
                │ │collect_│    │              │             │
                │ │results │    │              │             │
                │ └───┬────┘    │              │             │
                └─────┴─────────┴──────────────┴─────────────┘
                               │
                               ▼ (supervisor returns "FINISH")
                        ┌─────────────┐
                        │     END     │
                        └─────────────┘
```

A full cycle for a typical query looks like:

`START → supervisor → searcher → search_tools → collect_results → supervisor → summariser → supervisor → synthesiser → supervisor → END`

The supervisor may loop the searcher/summariser pair multiple times if the
first round of results is insufficient.

### State schema

`agents/state.py` defines `ResearchState`, a `TypedDict` with reducer-annotated
fields so worker nodes can append (rather than overwrite) results across loops:

- `query: str` — the user's research question.
- `messages: list[BaseMessage]` *(add_messages reducer)* — scratchpad used by
  the searcher / `ToolNode`.
- `search_results: list[SearchResult]` *(add reducer)* — normalised web hits.
- `summaries: list[str]` *(add reducer)* — one entry per summarisation batch.
- `final_answer: str` — populated by the Synthesiser.
- `next_agent: SupervisorDestination` — the supervisor's routing decision.
- `supervisor_notes: list[str]` *(add reducer)* — per-step routing rationale.
- `iteration: int` — guards against runaway supervisor loops.

---

## Why LangGraph (and not vanilla LangChain chains)?

A "supervisor delegates to workers, who report back, who get delegated to
again" pattern is awkward to express as a linear LangChain `Runnable | Runnable`
chain. LangGraph is a better fit for this problem because:

1. **Cyclic control flow.** The supervisor may invoke the searcher multiple
   times, or re-summarise after additional searches. Plain `RunnableSequence`
   pipelines are linear; expressing a loop requires manually orchestrating
   state outside the chain. LangGraph models cycles natively via conditional
   edges.
2. **Stateful coordination.** Every agent reads from and writes to a shared
   `TypedDict` state with reducer semantics (`add_messages`, `operator.add`),
   so workers don't need bespoke message-passing plumbing.
3. **Deterministic routing with LLM judgement.** Conditional edges let us mix
   structured LLM decisions (the supervisor picks the next node) with
   deterministic overrides (iteration caps, invariant guards) at the graph
   level rather than inside an agent's prompt.
4. **First-class tool execution.** `langgraph.prebuilt.ToolNode` handles the
   `AIMessage → ToolMessage` dance for free, so the Searcher agent only has
   to emit tool calls and the graph takes care of execution.
5. **Streaming and observability.** LangGraph exposes both `updates` and
   `messages` stream modes, so the CLI can print per-node progress *and*
   stream the Synthesiser's tokens in the same loop — something that would
   require manual callback wiring with vanilla chains.

For a single-shot RAG pipeline a linear chain is perfectly fine; for a
multi-agent assistant with conditional re-routing and shared state, LangGraph
removes a lot of accidental complexity.

---

## Project layout

```
.
├── agents/
│   ├── __init__.py
│   ├── state.py            # ResearchState TypedDict (typed state schema)
│   ├── supervisor.py       # Supervisor agent + routing function
│   ├── searcher.py         # Searcher agent + tool-result collector
│   ├── summariser.py       # Summariser agent
│   └── synthesiser.py      # Synthesiser agent (streamed)
├── tools/
│   ├── __init__.py
│   └── search.py           # Tavily / DuckDuckGo tool factory + normalisation
├── graph.py                # StateGraph wiring (ToolNode + conditional edges)
├── main.py                 # CLI entrypoint with token streaming
├── config.py               # .env loading + LLM/search provider factories
├── requirements.txt
├── .env.example
└── README.md
```

---

## Setup

1. **Clone and create a virtual environment**

   ```bash
   python -m venv .venv
   # Windows
   .venv\Scripts\activate
   # macOS / Linux
   source .venv/bin/activate
   ```

2. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

3. **Configure environment variables**

   ```bash
   cp .env.example .env
   ```

   Edit `.env`:

   - Set `LLM_PROVIDER` to `openai` or `anthropic` and provide the matching
     API key. The default models are `gpt-4o-mini` and
     `claude-3-5-sonnet-latest`; override with `OPENAI_MODEL` /
     `ANTHROPIC_MODEL`.
   - Set `SEARCH_PROVIDER` to `tavily` (recommended) and provide
     `TAVILY_API_KEY`. If the key is missing, the code automatically falls
     back to DuckDuckGo (no key required).
   - Tweak `MAX_SEARCH_RESULTS` and `SUPERVISOR_MAX_ITERATIONS` to taste.

---

## Usage

### Interactive REPL

```bash
python main.py
```

You'll be prompted for queries; each one is run through the full graph and
the final answer is streamed to the terminal.

### One-shot

```bash
python main.py "What are the latest advances in fusion ignition (2024-2025)?"
```

While the graph runs you'll see per-node progress lines such as:

```
→ Supervisor [step 1] -> searcher: No results yet; searching.
→ Searcher
→ Tool execution
→ Collecting results (+5 results)
→ Supervisor [step 2] -> summariser: Results collected; summarising.
→ Summariser (+1 summary batch)
→ Supervisor [step 3] -> synthesiser: Ready to synthesise.
→ Synthesiser (streaming...)

──────────────────────── Final answer ─────────────────────────
<streamed tokens here>
```

---

## Extending

- **Add a worker.** Implement a new node function in `agents/`, add a literal
  to `SupervisorDestination` in `agents/state.py`, register the node in
  `graph.py`, and update the Supervisor's system prompt so it knows when to
  call it.
- **Add a tool.** Add a factory in `tools/`, return it from
  `build_search_tool` (or build a new tool list), and pass it to either the
  Searcher's `bind_tools` call or a new `ToolNode`.
- **Persist runs.** Pass a checkpointer (e.g. `MemorySaver` or a SQLite
  checkpointer) into `workflow.compile(checkpointer=...)` in `graph.py` and
  supply a `thread_id` from the CLI to make sessions resumable.
