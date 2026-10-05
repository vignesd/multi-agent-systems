import asyncio
from langchain.messages import HumanMessage
from graph.workflow import build_workflow

from logging_config import setup_logging

setup_logging()

async def main(query: str):
    # 1. Compile the graph
    graph = await build_workflow()

    # 2. Define initial state matching the updated AgentState schema
    initial_state = {
        "messages": [HumanMessage(content=query)],
        "task_queue": [],
        "next_agent": "",
        "current_sub_task": None,
    }

    # 3. Invoke the workflow
    result = await graph.ainvoke(initial_state)

    # 4. Print final synthesized output
    print("\n==============================")
    print("FINAL RESPONSE")
    print("==============================\n")
    print(result["messages"][-1].content)


if __name__ == "__main__":
    test_prompts = [
        "Find flights on Nov 4, 2026 from Delhi to Chennai, Bangalore to France, and Tokyo to Singapore. and convert 200 dollars to indian, france  and london money",
    # ---------------------------------------------------------
    # 1. Single-Agent Test Prompts
    # ---------------------------------------------------------
    "What is the current local time in Tokyo, Japan?",
    "Can you find flights from Delhi to Tokyo for December 5, 2026?",
    "What is the current exchange rate from USD to JPY?",
    "Show me the score and verdict for the top Uniswap ETH/USDC liquidity pool.",
    # ---------------------------------------------------------
    # 2. Two-Agent Test Prompts
    # ---------------------------------------------------------
    "Find flights from London to New York for next week and convert the ticket prices from GBP to USD.",
    "What time is it in Sydney right now, and are there any direct flights from Singapore arriving there tomorrow?",
    "Get the current ETH/USD conversion rate and list recent ENTER/EXIT signals for the ETH liquidity pools.",
    "Find flights to Singapore for a crypto conference and check the current risk signals for Arbitrum DeFi pools.",
    # ---------------------------------------------------------
    # 3. Multi-Agent Test Prompts (3+ Agents)
    # ---------------------------------------------------------
    "I am planning a trip from Delhi to Tokyo on Dec 5, 2026. Check available flights, show the current time in Tokyo, and convert the flight cost from JPY to USD.",
    "Find flight options from New York to London, convert the price to JPY, and analyze the historical performance of top London-based DeFi pools.",
    "Check Tokyo time right now, find flights from Delhi to Tokyo for Dec 5, 2026, convert the flight fare to USD, and check the latest RISK_OFF signals for major DeFi liquidity pools.",
    # ---------------------------------------------------------
    # 4. Edge Case & Invalid Test Prompts
    # ---------------------------------------------------------
    "Can you write a Python script to perform binary search on an array?",
    "Who was the first president of the United States?",
    "What is the weather in Paris, and what is the current time there?",
    "How do i reset my phone password?"
]
    for question in test_prompts[2:4]:
         question=input("Enter your prompt -")
         print(f"Question - {question}")
         asyncio.run(main(question))
         break