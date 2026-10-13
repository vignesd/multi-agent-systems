# type:ignore
import asyncio

from langchain_core.messages import HumanMessage
from graph.workflow import build_workflow
from logging_config import setup_logging


setup_logging()


async def chat() -> None:
    """Run a continuous chat session with in-memory conversation history."""

    # Compile the workflow once and reuse it throughout the session.
    graph = await build_workflow()

    # Stores the conversation history while this process is running.
    conversation_history = []

    print("\n===================================")
    print("      AI MULTI-AGENT CHAT")
    print("===================================")
    print("Type 'exit' or 'quit' to stop.")
    print("Type 'clear' to clear conversation history.")
    print("===================================\n")

    while True:
        try:
            query = input("You: ").strip()

            if not query:
                continue

            # Exit the chat.
            if query.lower() in {"exit", "quit"}:
                print("\nChat session ended.")
                break

            # Clear previous conversation context.
            if query.lower() == "clear":
                conversation_history.clear()
                print("\nConversation history cleared.\n")
                continue

            # Build the initial state for this request.
            initial_state = {
                "messages": [
                    *conversation_history,
                    HumanMessage(content=query),
                ],
                "task_queue": [],
                "next_agent": "",
                "current_sub_task": None,
            }

            print("\nAssistant: Thinking...\n")
            result = await graph.ainvoke(initial_state)
            conversation_history = result["messages"]

            final_message = result["messages"][-1].content

            print("-----------------------------------")
            print("Assistant:")
            print(final_message)
            print("-----------------------------------\n")

        except KeyboardInterrupt:
            print("\n\nChat session interrupted.")
            break

        except Exception as exc:
            # Keep the chat session running if a request fails.
            print(f"\nError processing request: {exc}")
            print("You can try another prompt.\n")


if __name__ == "__main__":
    asyncio.run(chat())
