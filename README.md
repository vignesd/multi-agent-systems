# Multi-Agent AI System — LangGraph + MCP

A production-style multi-agent AI platform built with **LangChain v1**, **LangGraph**, and the **Model Context Protocol (MCP)**. A supervisor agent plans and decomposes user requests, routes sub-tasks to five domain specialist agents backed by live MCP servers, and synthesizes their results into a single answer — with bounded retries, domain guardrails, and a custom middleware layer for conversation summarization, PII masking, and prompt-injection defense.

Also included are two focused terminal assistants: a **Travel Assistant** (Kiwi MCP) and a **Chinook Database Assistant** that connects to an **API-key-authenticated MCP server**.

## Highlights

- **Supervisor–specialist orchestration (LangGraph):** a planner decomposes each request into typed sub-tasks using structured output (Pydantic schemas), executes them through specialist agents, and loops back until the queue is drained.
- **Five live MCP integrations:** Travel (Kiwi), Currency, World Clock, DeFi (Wealthville), and Web Search (Tavily) — all consumed via `langchain-mcp-adapters` over streamable HTTP.
- **Authenticated MCP access:** the Chinook agent sends an API key as a Bearer token to a protected remote MCP server, with explicit 401/HTTP error handling.
- **Custom agent middleware:**
  - Conversation **summarization** (triggers at 50 messages, keeps the latest 10, domain-specific summary prompts).
  - **PII masking** of email addresses and phone numbers in model output and tool results.
  - A **prompt-injection guard** (`@before_agent` hook) that rejects instruction-override attempts before the agent runs.
- **Resilience by design:** per-task retry budget (3 attempts), failed tasks reported honestly in the final answer — never invented.
- **Domain guardrails:** out-of-scope requests are rejected at the planning stage without invoking any specialist.
- **Observability:** structured logging with optional rotating file output, plus optional LangSmith tracing.

## Architecture

```mermaid
flowchart TD
    U([User]) --> S[Supervisor<br/>plan & route tasks]
    S -->|travel| T[Travel Agent<br/>Kiwi MCP]
    S -->|currency| C[Currency Agent<br/>Currency MCP]
    S -->|worldclock| W[World Clock Agent<br/>WorldClock MCP]
    S -->|defi| D[DeFi Agent<br/>Wealthville MCP]
    S -->|websearch| B[Web Search Agent<br/>Tavily MCP]
    T --> S
    C --> S
    W --> S
    D --> S
    B --> S
    S -->|all tasks done| F[Final Synthesis]
    F --> A([Unified Answer])
```

**How a request flows**

1. **Plan** — the supervisor extracts the latest user request (using prior turns only to resolve references like *"convert its price"*), and produces a validated task list via structured output. Unsupported requests are rejected immediately.
2. **Execute** — each pending task is routed to its specialist agent, which calls live MCP tools. Failures are retried up to 3 attempts, then marked failed with the error preserved.
3. **Synthesize** — a final node combines all successful results into one coherent response, clearly flagging anything that could not be completed.

**Standalone clients** (single-agent, independent of the workflow):

```text
clients/travel_client.py   -> Travel agent   -> Kiwi MCP server
clients/chinook_client.py  -> Chinook agent  -> API-key-protected Chinook MCP server
clients/coordinate.py      -> Plan-and-execute coordinator with concurrent task execution
```

## Project Structure

```text
.
├── agents/                  # Specialist agents + MCP tool wiring
│   ├── travel_agent.py      #   Kiwi MCP (flights, itineraries)
│   ├── currency_agent.py    #   Exchange rates & conversions
│   ├── worldclock_agent.py  #   Time zones, DST, city time
│   ├── defi_agent.py        #   Liquidity pools, scores, signals
│   ├── websearch_agent.py   #   Tavily search / extract / crawl / map
│   └── chinook_agent.py     #   Authenticated Chinook DB agent (full middleware stack)
├── clients/                 # Terminal chat clients
│   ├── travel_client.py     #   Travel-only assistant
│   ├── chinook_client.py    #   Chinook-only assistant
│   └── coordinate.py        #   Non-graph coordinator (concurrent execution)
├── graph/                   # LangGraph workflow
│   ├── workflow.py          #   Supervisor, specialist nodes, retries, synthesis
│   ├── state.py             #   Shared AgentState + task queue schema
│   └── models.py            #   Pydantic planning schemas & routing map
├── middleware/              # Summarization, PII masking, prompt-injection guard
├── prompts/                 # System & summarization prompts per domain
├── config.py                # Settings via pydantic-settings
├── logging_config.py        # Console + optional rotating-file logging
├── main.py                  # Multi-agent workflow chat (entry point)
└── pyproject.toml           # Dependencies (managed with uv)
```

## Tech Stack

| Layer | Technology |
| --- | --- |
| Orchestration | LangGraph (supervisor pattern, conditional routing) |
| Agent framework | LangChain v1 (`create_agent`, agent middleware) |
| Tool integration | MCP via `langchain-mcp-adapters` (streamable HTTP) |
| LLM | OpenAI (`gpt-4o-mini` by default, configurable) |
| Planning schemas | Pydantic structured output |
| Web search | Tavily MCP |
| Observability | Python logging + optional LangSmith tracing |
| Tooling | Python 3.13+, uv |

## Getting Started

### Prerequisites

- Python **3.13+**
- [uv](https://docs.astral.sh/uv/) package manager
- An OpenAI API key

### Installation

```bash
git clone https://github.com/vignesd/multi-agent-systems.git
cd multi-agent-systems
uv sync
```

### Configuration

Copy the example environment file and fill in your credentials:

```bash
cp .env.example .env        # macOS / Linux
Copy-Item .env.example .env # Windows PowerShell
```

| Variable | Required for | Description |
| --- | --- | --- |
| `OPENAI_API_KEY` | All clients | OpenAI API key for the chat model |
| `MODEL` | Optional | Chat model name (default: `gpt-4o-mini`) |
| `CHINOOK_MCP_SERVER_API_KEY` | Chinook assistant | Sent as a Bearer token to the protected Chinook MCP server |
| `TAVILY_API_KEY` | Multi-agent workflow | Used by the Web Search specialist |
| `LANGSMITH_API_KEY`, `LANGSMITH_TRACING` | Optional | Enable LangSmith tracing |
| `LOG_LEVEL`, `LOG_TO_FILE`, `LOG_FILE` | Optional | Logging configuration (file logging rotates at 10 MB) |

> Keep real credentials in your local `.env` only — never commit them.

## Usage

### Multi-agent workflow (LangGraph)

```bash
uv run python main.py
```

Initializes all five specialists, so their MCP services (including Tavily) must be reachable.

### Travel Assistant

```bash
uv run python -m clients.travel_client
```

### Chinook Database Assistant

```bash
uv run python -m clients.chinook_client
```

### Coordinator (non-graph, concurrent execution)

```bash
uv run python -m clients.coordinate
```

All clients run as interactive terminal chats. Type `clear` to reset the in-memory conversation, or `exit` / `quit` to end the session.

### Example prompts

**Multi-agent workflow**

- "Find a flight from Boston to Paris and convert its price to EUR."
- "What time is it in Tokyo, and what's the USD→JPY rate?"
- "Search for recent news about AI regulation in Europe."
- "Show me the top-rated DeFi liquidity pools and any recent risk signals."

**Chinook assistant**

- "Find the customer with email `luisg@embraer.com.br`."
- "Summarize the invoices for customer ID 5."

**Travel assistant**

- "Find flights from Boston to Paris next month."

## Middleware & Safety

The Chinook agent runs the full middleware stack; the Travel agent uses summarization.

| Middleware | What it does |
| --- | --- |
| Summarization | Compresses history once it reaches 50 messages, keeping the latest 10, using a domain-specific summary prompt |
| PII masking | Masks emails and phone numbers in assistant output and tool results |
| Prompt-injection guard | A `before_agent` hook that rejects common instruction-override patterns (e.g. *"ignore previous instructions"*, *"reveal your system prompt"*) before the agent executes |
| Domain guardrails | The workflow planner rejects out-of-scope requests at planning time, before any specialist or tool is invoked |

> The injection guard is a pattern-based first line of defense, not a complete security boundary — scope enforcement is reinforced in each agent's system prompt.

## Roadmap

- Automated test suite with `pytest` / `pytest-asyncio`
- Persistent conversation memory across sessions
- Checkpointing and human-in-the-loop interrupts in the LangGraph workflow
- Containerized deployment

## License

This project is licensed under the [MIT License](LICENSE).
