from __future__ import annotations

import asyncio
import logging
import threading
from collections.abc import Awaitable, Callable
from concurrent.futures import Future, TimeoutError as FutureTimeoutError
from dataclasses import dataclass
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage
from langchain_openai import ChatOpenAI

from utils.messages import get_final_answer

logger = logging.getLogger(__name__)

# A runner takes the full message list and returns the updated message list.
Runner = Callable[[list[Any]], Awaitable[list[Any]]]


@dataclass(frozen=True)
class ClientSpec:
    label: str
    icon: str
    description: str
    examples: tuple[str, ...]


CLIENTS: dict[str, ClientSpec] = {
    "chinook": ClientSpec(
        label="Chinook Customer Assistant",
        icon="🎵",
        description="Customer details, invoices and supported customer records.",
        examples=(
            "Show customer details for customer ID 10.",
            "Find a customer by email.",
            "Show a customer's invoices.",
        ),
    ),
    "travel": ClientSpec(
        label="Travel Assistant",
        icon="✈️",
        description="Flight search and travel options via the Kiwi MCP server.",
        examples=(
            "Find flights from London to Rome next Friday.",
            "Cheapest way to get from Paris to Berlin this weekend?",
        ),
    ),
    "workflow": ClientSpec(
        label="Multi-Agent Workflow",
        icon="🧭",
        description="Supervisor that plans and routes to travel, currency, world clock, DeFi and web search specialists.",
        examples=(
            "What time is it in Tokyo and what is 100 USD in JPY?",
            "Search the web for the latest news on LangGraph.",
        ),
    ),
}


async def _build_runner(client: str, model_name: str) -> Runner:
    # Imported lazily so missing credentials surface as request errors, not import errors.
    if client == "workflow":
        from graph.workflow import build_workflow

        graph = await build_workflow(model_name)

        async def run_workflow(messages: list[Any]) -> list[Any]:
            result = await graph.ainvoke({
                "messages": messages,
                "task_queue": [],
                "next_agent": "",
                "current_sub_task": None,
            })
            return list(result["messages"])

        return run_workflow

    model = ChatOpenAI(model=model_name, temperature=0.2)
    if client == "chinook":
        from agents.chinook_agent import create_customer_agent

        agent = await create_customer_agent(model)
    elif client == "travel":
        from agents.travel_agent import create_travel_agent

        agent = await create_travel_agent(model)
    else:
        raise ValueError(f"Unknown client: {client}")

    async def run_agent(messages: list[Any]) -> list[Any]:
        result = await agent.ainvoke({"messages": messages})
        returned = result.get("messages")
        if returned is not None:
            return list(returned)
        return [*messages, AIMessage(content=get_final_answer(result))]

    return run_agent


class AgentWorker:
    """Own one persistent event loop and a cache of built agents per (client, model)."""

    def __init__(self) -> None:
        self._loop = asyncio.new_event_loop()
        self._ready = threading.Event()
        self._runners: dict[tuple[str, str], Runner] = {}
        self._lock: asyncio.Lock | None = None

        self._thread = threading.Thread(
            target=self._run_loop,
            name="agent-loop",
            daemon=True,
        )
        self._thread.start()

        if not self._ready.wait(timeout=10):
            raise RuntimeError("Agent event loop failed to start.")

    def _run_loop(self) -> None:
        asyncio.set_event_loop(self._loop)
        self._lock = asyncio.Lock()
        self._ready.set()
        self._loop.run_forever()

    async def _get_runner(self, client: str, model_name: str) -> Runner:
        key = (client, model_name)
        async with self._lock:  # type: ignore[union-attr]
            if key not in self._runners:
                logger.info("Initializing %s agent with model %s", client, model_name)
                self._runners[key] = await _build_runner(client, model_name)
            return self._runners[key]

    async def _invoke(
        self,
        client: str,
        model_name: str,
        history: list[Any],
        query: str,
    ) -> tuple[str, list[Any]]:
        runner = await self._get_runner(client, model_name)
        messages = await runner([*history, HumanMessage(content=query)])
        return get_final_answer({"messages": messages}), messages

    def run_turn(
        self,
        client: str,
        model_name: str,
        history: list[Any],
        query: str,
        timeout: float = 180,
    ) -> tuple[str, list[Any]]:
        future: Future = asyncio.run_coroutine_threadsafe(
            self._invoke(client, model_name, history, query),
            self._loop,
        )
        try:
            return future.result(timeout=timeout)
        except FutureTimeoutError:
            future.cancel()
            raise TimeoutError(
                f"The request did not finish within {int(timeout)} seconds."
            ) from None
