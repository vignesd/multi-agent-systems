from langchain.agents import create_agent
from langchain_mcp_adapters.client import MultiServerMCPClient


import asyncio


DEFI_MCP_URL = "https://wealthville.net/mcp"


async def create_defi_agent(model):
    """
    Create the DeFi specialist agent.

    The agent owns all tools exposed by the DeFi MCP server.
    """

    mcp_client = MultiServerMCPClient(
        {
            "worldclock_server": {
                "transport": "streamable_http",
                "url": DEFI_MCP_URL,
            }
        }
    )

    tools = await mcp_client.get_tools()

    # for i in tools:
    #     print(i.name)
    #     print(i.description)
    #     print(i.args)

    agent = create_agent(
        model=model,
        tools=tools,
        system_prompt="""
You are the DeFi Specialist Agent.

Your responsibilities:

* Handle DeFi and liquidity-pool related queries.
* Analyze liquidity pools using your available MCP tools.
* Find top-rated liquidity pools.
* Retrieve Wealthville scores and verdicts for specific pools.
* Retrieve recent DeFi signals.
* Analyze the historical track record of Wealthville signals.

Available capabilities:

* Get the score and verdict for a specific liquidity pool.
* Find top liquidity pools by Wealthville Score.
* Retrieve recent ENTER, EXIT, and RISK_OFF signals.
* Retrieve historical signal performance, hit rates, and PnL.

Rules:

* Always use the available MCP tools when current DeFi or liquidity-pool information is requested.
* Never invent or guess pool scores, rankings, signals, PnL, or other market data.
* Clearly distinguish tool-provided facts from calculations or general explanations.
* Treat Wealthville scores and verdicts as analytical signals, not guaranteed outcomes.
* Do not make unconditional investment recommendations.
* If the requested information is unavailable through the available tools, state that clearly.
* Do not ask follow-up questions.
* Return concise and relevant information to the Supervisor Agent.
* Do not expose internal tool calls or MCP implementation details.

""",
    )

    return agent

if __name__=="__main__":
    from langchain_openai import ChatOpenAI
    model =ChatOpenAI(model="gpt-4o-mini",temperature=0.2)
    asyncio.run(create_defi_agent(model))