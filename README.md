# Multi-Agent Systems

A Python-based multi-agent orchestration project built with LangGraph and LangChain. The system routes a user request to specialist agents for travel, currency conversion, world clock, DeFi, and web search, then combines the results into a single final response.

## Overview

This project demonstrates a supervisor-driven agent architecture:

- A supervisor agent decomposes each user request into focused sub-tasks.
- Specialist agents handle domain-specific work using external MCP tool servers.
- A final synthesis step combines the collected results into one natural answer.
- Failed specialist tasks are retried with bounded attempts before being reported honestly.

The supported domains are:

- Travel planning and flight search
- Currency exchange and conversion
- World clock and timezone conversion
- DeFi / liquidity pool analysis
- Web search and extraction

## Architecture

The project is organized into a small set of core components:

- `main.py` — interactive CLI entry point for chat-style requests.
- `config.py` — environment and logging configuration.
- `graph/workflow.py` — LangGraph workflow, supervisor logic, task queue, retries, and final synthesis.
- `graph/models.py` — structured schemas for agent planning and task definitions.
- `graph/state.py` — typed workflow state for the orchestration graph.
- `agents/` — specialist agent implementations for each domain.

### Workflow flow

1. The CLI sends the user message into the LangGraph workflow.
2. The supervisor plans one or more sub-tasks based on the query.
3. For each task, a domain specialist agent executes the work using MCP-backed tools.
4. Results are appended to the state and routed back to the supervisor.
5. Once all tasks are handled, a final agent compiles the response.

## Specialist agents

The project includes domain-specific specialist agents: 

- `travel_agent.py` — flight and travel information using the Kiwi MCP server.
- `currency_agent.py` — exchange-rate conversion using the currency MCP service.
- `worldclock_agent.py` — current time and timezone queries using the World Clock MCP service.
- `defi_agent.py` — DeFi signal and liquidity pool analysis using the Wealthville MCP service.
- `websearch_agent.py` — web search and content extraction using Tavily MCP capabilities.

These agents are created as LangChain tool-using agents that use live MCP tools rather than static prompts alone.

## Project structure

```text
.
├── README.md
├── config.py
├── coordinate.py
├── logging_config.py
├── main.py
├── pyproject.toml
├── travel_client.py
├── uv.lock
├── agents/
│   ├── __init__.py
│   ├── currency_agent.py
│   ├── defi_agent.py
│   ├── travel_agent.py
│   ├── websearch_agent.py
│   └── worldclock_agent.py
├── graph/
│   ├── __init__.py
│   ├── models.py
│   ├── state.py
│   └── workflow.py
├── notebooks/
│   └── practice.ipynb
└── .env (local environment file, not tracked)
```

## Prerequisites

- Python 3.13+
- An OpenAI-compatible model available through LangChain/OpenAI
- API keys for the services used by the agents
- `uv` for dependency management (recommended)

## Getting started

1. Clone the repository:

```bash
git clone https://github.com/vignesd/multi-agent-systems.git
cd multi-agent-systems
```

2. Create and configure your environment variables in a `.env` file:

```env
OPENAI_API_KEY=your_openai_api_key
MODEL=gpt-4o-mini
TAVILY_API_KEY=your_tavily_api_key
```

3. Install dependencies:

```bash
uv sync
```

4. Run the interactive chat app:

```bash
uv run python main.py
```

## Usage

Once the app starts, you can ask questions such as:

- "What is the current exchange rate from USD to EUR?"
- "Find me flights from New York to London next week."
- "What time is it in Tokyo right now?"
- "Compare the best DeFi pools with liquidity signals."
- "Search for recent updates on AI regulation in Europe."

The assistant will reason over the request, assign relevant tasks, and return a synthesized response.

## Notes

- The workflow enforces supported domain checks and rejects unsupported requests.
- Specialist agents are designed to rely on external tool data rather than guessing.
- For web search, the Tavily API key must be present.
- For model access, the OpenAI API key must be available in the environment.

## Example flow

```text
User: Find the best flight from Boston to Paris in two weeks and convert the estimated price to EUR.

Supervisor:
- travel: search flight options and provide fare details
- currency: convert fare amount to EUR

Final answer:
- Summarized flight choices
- Currency-converted total in EUR
```

## License

This project does not currently declare a license in the repository.

## Contributing

Contributions are welcome. Please keep changes focused, document new agents or workflow changes, and validate behavior with the existing test setup when available.
