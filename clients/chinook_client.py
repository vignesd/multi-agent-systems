
from __future__ import annotations

import asyncio
import logging
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage
from langchain_openai import ChatOpenAI

# from deepagents.middleware.subagents import SubAgentMiddleware

from agents.chinook_agent import create_customer_agent
from config import MODEL
from utils.messages import get_final_answer
from logging_config import setup_logging


logger = logging.getLogger(__name__)

EXIT_COMMANDS = {"exit", "quit"}
CLEAR_COMMAND = "clear"


async def run_customer_chat() -> None:
    """Run a continuous terminal chat using only the chinook customer agent."""

    setup_logging()

    logger.info("Starting chinook customer client with model: %s", MODEL)

    # Initialize the model and ONLY the chinook customer agent.
    model = ChatOpenAI(
        model=MODEL,
        temperature=0.2,
    )

    try:
        customer_agent = await create_customer_agent(model)

    except Exception:
        logger.exception("Failed to initialize the customer agent")
        print(
            "\nCould not initialize the chinook customer agent. "
            "Check your API key, MCP server connection, and logs."
        )
        return

    # In-memory session history. It lasts until the program exits
    # or the user enters "clear".
    conversation_history: list[Any] = []

    print("\nchinook customer Assistant is ready.")
    print(f"Model: {MODEL}")
    print("Available commands: clear, exit, quit")
    print("Ask about customer details by ID/Email or Customer Invoices or other supported chinook customer tasks.")
    print("-" * 60)

    while True:
        try:
            query = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nEnding chinook customer session.")
            break

        if not query:
            continue

        command = query.lower()

        if command in EXIT_COMMANDS:
            print("chinook customer session ended.")
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
            result = await customer_agent.ainvoke(
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

            print(f"\nchinook Assistant:\n{answer}")
            # print(f"Converstation len - {len(conversation_history)}")
            # if len(conversation_history) >= 10:
            #     print(f"Summary - {conversation_history[0]}")

        except Exception:
            logger.exception("chinook customer request failed")
            print(
                "\nSorry, the chinook customer request failed. "
                "Check the logs and try again."
            )


if __name__ == "__main__":
    asyncio.run(run_customer_chat())
