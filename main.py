import asyncio
from langchain.messages import HumanMessage
from graph.workflow import build_workflow

async def main(query:str):
    graph = await build_workflow()
#     query = """
# Find 2 available flights from Dubai to London
# for December 5, 2026.

# For each flight:
# - Show airline
# - Show flight number
# - Show departure time
# - Show arrival time
# - Show price
# - Convert the price to INR
# - Tell me the current time in London
# """

    result = await graph.ainvoke(
        {
            "messages": [
                HumanMessage(content=query)
            ],
            "next_agent": "",
            "travel_result": "",
            "currency_result": "",
            "worldclock_result": "",
        }
    )

    print("\n==============================")
    print("FINAL RESPONSE")
    print("==============================\n")

    print(result["messages"][-1].content)


if __name__ == "__main__":
    test_prompts = [
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
]
    for question in test_prompts[8:9]:
         print(f"Question - {question}")
         question="""
         Find a flight on 4th nov 26 for
            From Delhi to chennai
            Bangalore to France and Tokoyo to singapore 
         """
         asyncio.run(main(question))
         break