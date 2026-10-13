
from __future__ import annotations

import asyncio
import logging
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage
from langchain_openai import ChatOpenAI

from agents.travel_agent import create_travel_agent
from config import MODEL
from logging_config import setup_logging


logger = logging.getLogger(__name__)

EXIT_COMMANDS = {"exit", "quit"}
CLEAR_COMMAND = "clear"


def extract_text(content: Any) -> str:
    """Convert LangChain message content into printable text."""

    if isinstance(content, str):
        return content.strip()

    if isinstance(content, list):
        parts: list[str] = []

        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict):
                # Handle common text content blocks.
                if block.get("type") == "text":
                    text = block.get("text")
                    if isinstance(text, str):
                        parts.append(text)

        return "\n".join(parts).strip()

    return str(content).strip() if content is not None else ""


def get_final_answer(result: dict[str, Any]) -> str:
    """Extract the latest assistant response from an agent result."""

    messages = result.get("messages", [])

    for message in reversed(messages):
        if isinstance(message, AIMessage):
            # Skip assistant messages that only contain tool calls.
            if message.tool_calls:
                continue

            answer = extract_text(message.content)
            if answer:
                return answer

    return "The travel agent did not return a text response."


async def run_travel_chat() -> None:
    """Run a continuous terminal chat using only the travel agent."""

    setup_logging()

    logger.info("Starting travel client with model: %s", MODEL)

    # Initialize the model and ONLY the travel agent.
    model = ChatOpenAI(
        model=MODEL,
        temperature=0.2,
    )

    try:
        travel_agent = await create_travel_agent(model)
    except Exception:
        logger.exception("Failed to initialize the travel agent")
        print(
            "\nCould not initialize the travel agent. "
            "Check your API key, MCP server connection, and logs."
        )
        return

    # In-memory session history. It lasts until the program exits
    # or the user enters "clear".
    conversation_history: list[Any] = []

    print("\nTravel Assistant is ready.")
    print(f"Model: {MODEL}")
    print("Available commands: clear, exit, quit")
    print("Ask about flights, routes, fares, or other supported travel tasks.")
    print("-" * 60)

    while True:
        try:
            query = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nEnding travel session.")
            break

        if not query:
            continue

        command = query.lower()

        if command in EXIT_COMMANDS:
            print("Travel session ended.")
            break

        if command == CLEAR_COMMAND:
            conversation_history.clear()
            print("Conversation history cleared.")
            continue

        # Pass earlier messages so the agent can answer follow-ups.
        messages = [
            *conversation_history,
            HumanMessage(content=query),
        ]

        try:
            result = await travel_agent.ainvoke(
                {"messages": messages}
            )

            answer = get_final_answer(result)

            # Preserve the agent's complete message history, including
            # tool interactions, for follow-up questions.
            returned_messages = result.get("messages")
            if returned_messages is not None:
                conversation_history = list(returned_messages)
            else:
                # Defensive fallback for an unexpected result shape.
                conversation_history.extend(
                    [
                        HumanMessage(content=query),
                        AIMessage(content=answer),
                    ]
                )

            print(f"\nTravel Assistant:\n{answer}")

        except Exception:
            logger.exception("Travel request failed")
            print(
                "\nSorry, the travel request failed. "
                "Check the logs and try again."
            )


if __name__ == "__main__":
    asyncio.run(run_travel_chat())
