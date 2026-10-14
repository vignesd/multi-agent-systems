

from __future__ import annotations

import asyncio
import logging
from typing import Any

from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    SystemMessage,
)
from langchain_openai import ChatOpenAI

from agents.currency_agent import create_currency_agent
from agents.defi_agent import create_defi_agent
from agents.travel_agent import create_travel_agent
from agents.websearch_agent import create_websearch_agent
from agents.worldclock_agent import create_worldclock_agent

from config import MODEL
from graph.models import InitialPlan
from logging_config import setup_logging


logger = logging.getLogger(__name__)

SUPPORTED_AGENTS = {
    "travel",
    "currency",
    "worldclock",
    "defi",
    "websearch",
}

MAX_ATTEMPTS = 3
MAX_CONTEXT_MESSAGES = 10
MAX_CONCURRENCY = 5


PLANNER_PROMPT = """
You are an expert task-planning agent coordinating specialist agents.

SUPPORTED DOMAINS:

1. travel: Flight searches, itineraries, travel information.
2. currency: Exchange rates and monetary conversions.
3. worldclock: Current time and time-zone calculations.
4. defi: Crypto liquidity pools and risk/entry/exit signals.
5. websearch: Current news, facts, web searches, URL extraction.

DOMAIN RULES:

- All five domains are supported.
- Requests can involve multiple supported domains.
- Reject requests outside these domains.
- Never reject a request just because it needs multiple agents.
- Never generate dummy, filler, or N/A tasks.
- Valid agent names are exactly:
  travel, currency, worldclock, defi, websearch.

CONVERSATION RULES:

- Plan tasks only for the current user request.
- Use previous conversation to resolve references such as
  "there", "that flight", "its price", or "convert it".
- Do not repeat earlier tasks unless explicitly requested.
- Make every task query clear and self-contained.
- Include relevant previous results in follow-up task queries.
- Do not invent missing prices, rates, or facts.
- Prefer independent tasks when multiple agents are required.
"""


def content_to_text(content: Any) -> str:
    """Convert message content into readable text."""
    if isinstance(content, str):
        return content

    return str(content)


def extract_agent_response(result: Any) -> str:
    """Extract a text response from a specialist agent."""
    if isinstance(result, dict):
        messages = result.get("messages") or []

        if messages:
            return content_to_text(messages[-1].content)

        if "output" in result:
            return content_to_text(result["output"])

    return content_to_text(result)


def format_history(messages: list[Any]) -> str:
    """Format recent conversation turns for the planner."""
    lines = []

    for message in messages[-MAX_CONTEXT_MESSAGES:]:
        if isinstance(message, HumanMessage):
            role = "User"
        elif isinstance(message, AIMessage):
            role = "Assistant"
        else:
            continue

        content = content_to_text(message.content).strip()

        if content:
            lines.append(f"{role}: {content}")

    return "\n".join(lines) or "No previous conversation."


def validate_plan(plan: InitialPlan) -> list[dict[str, str]]:
    """Validate and deduplicate tasks before dispatching them."""
    valid_tasks = []
    seen_tasks = set()

    for item in plan.tasks or []:
        query = (item.query or "").strip()
        agent_name = (item.agent or "").strip().lower()

        if not query or query.upper() in {"N/A", "NONE", "NULL"}:
            logger.warning("Ignoring empty or placeholder task.")
            continue

        if agent_name not in SUPPORTED_AGENTS:
            logger.warning(
                "Ignoring unsupported agent: %s",
                agent_name,
            )
            continue

        task_key = (agent_name, query.casefold())

        if task_key in seen_tasks:
            logger.warning("Ignoring duplicate task: %s", query)
            continue

        seen_tasks.add(task_key)

        valid_tasks.append({
            "agent": agent_name,
            "query": query,
        })

    return valid_tasks


async def build_coordinator() -> dict[str, Any]:
    """Initialize the planner and specialist agents once per session."""
    model = ChatOpenAI(
        model=MODEL,
        temperature=0.0,
    )

    planner = model.with_structured_output(InitialPlan)

    logger.info("Initializing specialist agents...")

    (
        travel_agent,
        currency_agent,
        worldclock_agent,
        defi_agent,
        websearch_agent,
    ) = await asyncio.gather(
        create_travel_agent(model),
        create_currency_agent(model),
        create_worldclock_agent(model),
        create_defi_agent(model),
        create_websearch_agent(model),
    )

    agents = {
        "travel": travel_agent,
        "currency": currency_agent,
        "worldclock": worldclock_agent,
        "defi": defi_agent,
        "websearch": websearch_agent,
    }

    logger.info(
        "Initialized agents: %s",
        ", ".join(agents.keys()),
    )

    return {
        "model": model,
        "planner": planner,
        "agents": agents,
    }


async def execute_task(
    task: dict[str, str],
    agent: Any,
    semaphore: asyncio.Semaphore,
) -> dict[str, Any]:
    """Execute one specialist task with a maximum of three attempts."""
    agent_name = task["agent"]
    query = task["query"]
    last_error = "Unknown error"

    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            async with semaphore:
                logger.info(
                    "Executing %s | Attempt %s/%s",
                    agent_name,
                    attempt,
                    MAX_ATTEMPTS,
                )

                result = await agent.ainvoke({
                    "messages": [
                        HumanMessage(content=query)
                    ]
                })

            response = extract_agent_response(result).strip()

            if not response:
                raise RuntimeError(
                    "The specialist returned an empty response."
                )

            logger.info(
                "%s completed successfully.",
                agent_name,
            )

            return {
                **task,
                "status": "completed",
                "attempts": attempt,
                "response": response,
                "error": None,
            }

        except Exception as exc:
            last_error = f"{type(exc).__name__}: {exc}"

            logger.exception(
                "%s failed | Attempt %s/%s",
                agent_name,
                attempt,
                MAX_ATTEMPTS,
            )

            if attempt < MAX_ATTEMPTS:
                # Backoff: 1 second, then 2 seconds.
                await asyncio.sleep(2 ** (attempt - 1))

    logger.error(
        "%s failed after %s attempts.",
        agent_name,
        MAX_ATTEMPTS,
    )

    return {
        **task,
        "status": "failed",
        "attempts": MAX_ATTEMPTS,
        "response": "",
        "error": last_error,
    }


async def process_turn(
    query: str,
    history: list[Any],
    coordinator: dict[str, Any],
) -> str:
    """Plan and execute one chat turn."""
    model = coordinator["model"]
    planner = coordinator["planner"]
    agents = coordinator["agents"]

    logger.info("=" * 70)
    logger.info("CURRENT USER REQUEST: %s", query)

    planner_input = [
        SystemMessage(content=PLANNER_PROMPT),
        HumanMessage(
            content=(
                "PREVIOUS CONVERSATION:\n"
                f"{format_history(history)}\n\n"
                "CURRENT USER REQUEST:\n"
                f"{query}"
            )
        ),
    ]

    # PHASE 1: Planning.
    try:
        plan: InitialPlan = await planner.ainvoke(planner_input)
        logger.info("Planner output: %s", plan.model_dump())

    except Exception as exc:
        logger.exception("Planner failed.")

        return (
            "Unable to plan this request because the planner failed: "
            f"{type(exc).__name__}: {exc}"
        )

    if not plan.is_supported:
        reason = (
            plan.rejection_reason
            or "The request is outside the supported domains."
        )

        logger.warning("Request rejected: %s", reason)

        return f"I cannot handle that request. {reason}"

    tasks = validate_plan(plan)

    if not tasks:
        return (
            "No valid tasks were generated for the supported domains: "
            "travel, currency, world clock, DeFi, and web search."
        )

    logger.info("Validated %s task(s).", len(tasks))

    for index, task in enumerate(tasks, start=1):
        logger.info(
            "Task %s: [%s] %s",
            index,
            task["agent"].upper(),
            task["query"],
        )

    # PHASE 2: Execute tasks concurrently.
    semaphore = asyncio.Semaphore(MAX_CONCURRENCY)

    results = await asyncio.gather(
        *[
            execute_task(
                task=task,
                agent=agents[task["agent"]],
                semaphore=semaphore,
            )
            for task in tasks
        ]
    )

    # PHASE 3: Synthesize the results.
    results_text = "\n\n".join(
        (
            f"Agent: {result['agent']}\n"
            f"Task: {result['query']}\n"
            f"Status: {result['status']}\n"
            f"Result: "
            f"{result['response'] or result['error']}"
        )
        for result in results
    )

    synthesis_prompt = SystemMessage(
        content="""
You are the final response coordinator.

Answer the current user request using the specialist results.

Rules:
- Address each part of the request.
- Clearly identify failed tasks or missing information.
- Never invent current times, exchange rates, flight availability,
  prices, news, or DeFi data.
- Use clear headings for multi-part answers.
- Do not claim a task succeeded if its status is failed.
- Do not mention internal retries or orchestration details.
"""
    )

    synthesis_input = [
        synthesis_prompt,
        HumanMessage(
            content=(
                f"CURRENT USER REQUEST:\n{query}\n\n"
                f"SPECIALIST RESULTS:\n{results_text}"
            )
        ),
    ]

    try:
        final_message = await model.ainvoke(synthesis_input)
        answer = content_to_text(final_message.content).strip()

    except Exception:
        logger.exception("Final synthesis failed.")

        answer = (
            "I couldn't combine the results into a single response. "
            "Here are the specialist results:\n\n"
            f"{results_text}"
        )

    logger.info("Turn completed.")
    return answer


async def chat() -> None:
    """Run continuous chat with session-only conversation memory."""
    setup_logging()

    coordinator = await build_coordinator()
    history: list[Any] = []

    print("\n" + "=" * 65)
    print("MULTI-AGENT COORDINATOR")
    print("=" * 65)
    print("Domains: travel, currency, worldclock, defi, websearch")
    print("Type 'clear' to reset memory.")
    print("Type 'exit' or 'quit' to stop.")
    print("=" * 65)

    while True:
        try:
            query = input("\nYou: ").strip()

        except (EOFError, KeyboardInterrupt):
            print("\nChat session ended.")
            break

        if not query:
            continue

        if query.lower() in {"exit", "quit"}:
            print("Chat session ended.")
            break

        if query.lower() == "clear":
            history.clear()
            print("Conversation history cleared.")
            continue

        try:
            answer = await process_turn(
                query=query,
                history=history,
                coordinator=coordinator,
            )

            print(f"\nAssistant:\n{answer}")

            # Preserve the user request and synthesized response.
            history.append(HumanMessage(content=query))
            history.append(AIMessage(content=answer))

        except Exception as exc:
            logger.exception("Unexpected error during this turn.")

            print(
                "\nAn unexpected error occurred: "
                f"{type(exc).__name__}: {exc}"
            )
            print("You can try another request.")


if __name__ == "__main__":
    asyncio.run(chat())
