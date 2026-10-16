# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

Python 3.13+, managed with `uv`. No linter is configured. Offline pytest tests live in `tests/` (note: `.gitignore` currently ignores `tests*`).

```bash
uv sync                                    # install deps
cp .env.example .env                       # then fill in keys (Copy-Item on PowerShell)
uv run python main.py                      # multi-agent LangGraph workflow (needs all 5 MCP services + OPENAI_API_KEY, TAVILY_API_KEY)
uv run python -m clients.travel_client     # travel-only assistant (Kiwi MCP)
uv run python -m clients.chinook_client    # Chinook DB assistant (needs CHINOOK_MCP_SERVER_API_KEY)
uv run python -m clients.coordinate        # non-graph plan-and-execute coordinator
uv sync --extra ui                         # adds Streamlit
uv run streamlit run ui/app.py             # Streamlit UI: pick client (Chinook / Travel / Workflow) and model in the sidebar
uv run pytest tests -v                     # offline middleware tests (fake models, no network)
uv run pytest tests/test_middleware.py::test_injection_guard_ends_run_without_calling_model   # single test
```

Clients are run as modules (`-m clients.x`) so top-level imports (`agents`, `config`, `middleware`, `prompts`) resolve.

## Architecture

LangChain v1 (`create_agent`) + LangGraph + MCP (`langchain-mcp-adapters`, streamable HTTP). Every specialist agent is built in `agents/*_agent.py` by connecting a `MultiServerMCPClient` to a remote MCP server, fetching its tools, and wrapping them in `create_agent` with a system prompt from `prompts/system_prompts.py`.

- `graph/workflow.py` (`build_workflow`): supervisor pattern. `supervisor` node plans via Pydantic structured output (`graph/models.py`, which also holds the task-type → node routing map), conditional edges route to one specialist node (`travel`, `currency`, `worldclock`, `defi`, `websearch`), each of which edges back to `supervisor`; when the task queue is empty it goes to `final` (synthesis) → END. State is `graph/state.py:AgentState`. `execute_specialist` implements the 3-attempt retry; failed tasks are carried into the final answer rather than invented. Out-of-scope requests are rejected in planning.
- `clients/coordinate.py`: an alternative, non-graph coordinator that executes planned tasks concurrently (it replaced an earlier `workflow_1.py`).
- `middleware/agent_middleware.py`: `get_default_middleware(model)` is the shared reliability stack (call limits, model fallback/retry, context editing, tool audit/error/retry) and every agent spreads it into `create_agent(middleware=[...])`. Per-agent extras: summarization (50 msgs trigger, keep 10), email/phone PII masking, and a `@before_agent(can_jump_to=["end"])` prompt-injection guard, which Chinook puts first. Middleware list order is outermost → innermost; `ToolErrorMiddleware` must stay outside `ToolRetryMiddleware(on_failure="error")`.
- Testing with `GenericFakeChatModel`: set `LANGSMITH_TRACING=false` (`tests/conftest.py` does this before imports). With tracing on (the `.env` default), the tracer uses up the fake model's message iterator and causes misleading `StopIteration`/`RuntimeError` failures.
- `agents/chinook_agent.py`: authenticates to the Chinook MCP server with a Bearer token read from `CHINOOK_MCP_SERVER_API_KEY` when the agent is created; connection/401 errors are re-raised.
- `config.py` holds `MODEL` and `AVAILABLE_MODELS` (the UI model list) plus pydantic-settings; `logging_config.py` sets up console + optional rotating-file logging. `utils/messages.py` has the shared `content_to_text` / `get_final_answer` helpers used by the CLI clients and the UI.
- `ui/`: Streamlit front end. `chat_service.py` defines `CLIENTS` (the client registry) and `AgentWorker`, which runs all agents on one background asyncio loop and caches a built agent per `(client, model)`. `app.py` keeps one `AgentWorker` per process (`st.cache_resource`) and a separate conversation history per client in `session_state`. To add a client to the UI, add a `ClientSpec` to `CLIENTS` and a branch in `_build_runner`.
- Root `clients.py` is a small launcher that has the same name as the `clients/` package (the package wins on import, so `clients.py` is only usable as a script); prefer `-m clients.<name>`.
