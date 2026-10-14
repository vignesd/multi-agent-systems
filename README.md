# Multi-Agent Systems

A Python project exploring tool-using AI agents, MCP integrations, middleware, and multi-agent orchestration. It includes two focused terminal assistants—a Travel Assistant and a Chinook database assistant—and a LangGraph workflow that can coordinate specialist agents across several domains.

## What it does

- **Travel Assistant:** uses the Kiwi MCP server for live flight and travel queries.
- **Chinook Assistant:** connects to an API-key-protected Chinook database MCP server and answers questions using the tools it exposes.
- **Multi-agent workflow:** plans and routes requests to Travel, Currency, World Clock, DeFi, and Web Search specialists, then synthesizes their results. Supported tasks can be retried, and incomplete results are reported rather than invented.
- **Conversation middleware:** summarizes longer chats, masks detected email addresses and phone numbers in assistant output, and blocks a set of common prompt-injection phrases in the Chinook assistant.

The focused clients are useful for exploring one integration at a time. The LangGraph workflow is the entry point for requests that may need more than one specialist.

## Architecture

```text
Focused clients
  clients/travel_client.py  -> Travel agent  -> Kiwi MCP server
  clients/chinook_client.py -> Chinook agent -> Authenticated Chinook MCP server

Orchestrated workflow
  main.py -> LangGraph supervisor -> specialist agent(s) -> final response
                                      ├─ Travel
                                      ├─ Currency
                                      ├─ World Clock
                                      ├─ DeFi
                                      └─ Web Search
```

The workflow plans tasks for the current request, executes them through domain-specific agents, and combines successful results. It makes up to three attempts for a specialist task before marking it incomplete.

## Project layout

```text
.
├── agents/                 # Domain-specific agent and MCP tool setup
├── clients/                # Independent Travel and Chinook terminal clients
├── graph/                  # LangGraph workflow, state, and planning schemas
├── middleware/             # Conversation summarization, PII, and prompt guard
├── prompts/                # Agent and summarization prompts
├── notebooks/              # Development notebook
├── clients.py              # Chinook client convenience entry point
├── config.py               # Model and application settings
├── logging_config.py       # Console and optional rotating-file logging
├── main.py                 # Multi-agent workflow chat entry point
├── pyproject.toml          # Project metadata and dependencies
└── uv.lock                 # Locked dependency versions
```

## Requirements

- Python 3.13 or newer
- [uv](https://docs.astral.sh/uv/) for dependency and virtual-environment management
- An OpenAI API key for the configured chat model
- Additional service credentials depending on the client or workflow you run

## Setup

1. Clone the repository and enter the project directory:

   ```bash
   git clone https://github.com/vignesd/multi-agent-systems.git
   cd multi-agent-systems
   ```

2. Install the project dependencies:

   ```bash
   uv sync
   ```

3. Copy `.env.example` to `.env` in the project root, then replace its placeholder values with your own credentials. On Windows PowerShell, use `Copy-Item .env.example .env`; on macOS or Linux, use `cp .env.example .env`.

   Set `OPENAI_API_KEY` and, as needed, add or update these values:

   | Variable | Required for | Description |
   | --- | --- | --- |
   | `OPENAI_API_KEY` | All clients | OpenAI API key used by the chat model. |
   | `MODEL` | Optional | Model name; defaults to `gpt-4o-mini`. |
   | `CHINOOK_MCP_SERVER_API_KEY` | Chinook client | API key for the configured Chinook MCP server. |
   | `TAVILY_API_KEY` | Full multi-agent workflow | Tavily key used by the Web Search specialist. |
   | `LANGSMITH_API_KEY` and `LANGSMITH_TRACING` | Optional tracing | Configure LangSmith tracing if desired. |

`MODEL` is optional; the default is `gpt-4o-mini`. Keep real credentials in your local `.env` file and never commit them. The Chinook API key is sent as a Bearer token to the configured MCP server.

## Run the assistants

### Travel Assistant

```bash
uv run python -m clients.travel_client
```

### Chinook database Assistant

```bash
uv run python -m clients.chinook_client
```

The root-level `clients.py` file is also a Chinook-only convenience entry point:

```bash
uv run python clients.py
```

### Multi-agent workflow

```bash
uv run python main.py
```

The workflow initializes all five specialist agents, so its external MCP services—including Tavily for Web Search—must be reachable and correctly configured before the chat can start.

Each chat runs in the terminal. Enter `clear` to clear the current in-memory conversation or `exit` / `quit` to end the session. Conversation history is not persisted after the process exits.

## Example requests

**Travel client**

- “Find flights from Boston to Paris next month.”
- “What are the available travel options from New York to London?”

**Chinook client**

- “Find the customer with this email address.”
- “Summarize the invoices for customer ID 5.”
- “Search the database for tracks by this artist.”

The Chinook assistant can only answer questions supported by the MCP tools enabled by the connected server.

**Multi-agent workflow**

- “What is the current exchange rate from USD to EUR?”
- “What time is it in Tokyo?”
- “Search for recent news about AI regulation in Europe.”
- “Find a flight from Boston to Paris and convert its price to EUR.”
- “Compare recent DeFi signals with current information from the web.”

Live data, available operations, and response quality depend on the connected MCP services and configured model.

## Middleware and conversation behavior

- The conversation summarizer is configured to summarize when the message history reaches 50 messages and retain the most recent 10 messages.
- The Chinook client masks detected email addresses in assistant output and phone numbers in assistant output and tool results.
- The Chinook prompt-injection guard rejects a defined set of common instruction-override patterns before agent execution. It is a basic safeguard, not a general-purpose security boundary.

## Configuration and logging

Application and logging settings can be provided through environment variables. The main settings include `APP_ENV`, `LOG_LEVEL`, `LOG_TO_FILE`, `LOG_FILE`, `LOG_MAX_BYTES`, and `LOG_BACKUP_COUNT`. File logging is disabled by default; when enabled, logs rotate at the configured size.

## Development

`pytest` and `pytest-asyncio` are included in the project dependencies, but no automated test suite is currently checked in.

## License

No license is currently declared in this repository.
