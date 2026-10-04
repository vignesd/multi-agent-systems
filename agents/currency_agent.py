from langchain.agents import create_agent
from langchain_mcp_adapters.client import MultiServerMCPClient


CURRENCY_MCP_URL = "https://currency-mcp.wesbos.com/mcp"


async def create_currency_agent(model):
    """
    Create the Currency specialist agent.

    The agent owns all tools exposed by the currency MCP server.
    """

    mcp_client = MultiServerMCPClient(
        {
            "currency_server": {
                "transport": "streamable_http",
                "url": CURRENCY_MCP_URL,
            }
        }
    )

    tools = await mcp_client.get_tools()

    agent = create_agent(
        model=model,
        tools=tools,
        system_prompt="""
You are the Currency Specialist Agent.

Your responsibilities:
- Currency conversion.
- Current exchange rates.
- Historical exchange rates.
- Currency-related calculations.

Rules:
- Always use your MCP tools for exchange-rate information.
- Never guess or invent exchange rates.
- Pay attention to the source and target currencies.
- If another agent provides an amount and currency, use those exact values.
- Do not ask follow-up questions.
- Return concise information to the supervisor.
""",
    )

    return agent
