from langchain.agents import create_agent
from langchain_mcp_adapters.client import MultiServerMCPClient

TRAVEL_MCP_URL = "https://mcp.kiwi.com"


async def create_travel_agent(model):
    """
    Create the Travel specialist agent.

    The agent owns all tools exposed by the Kiwi MCP server.
    """

    mcp_client = MultiServerMCPClient(
        {
            "travel_server": {
                "transport": "streamable_http",
                "url": TRAVEL_MCP_URL,
            }
        }
    )

    tools = await mcp_client.get_tools()

    agent = create_agent(
        model=model,
        tools=tools,
        system_prompt="""
You are the Travel Specialist Agent.

Your responsibilities:
- Search for flights.
- Find travel options.
- Handle flight-related information.
- Return accurate travel information using your available MCP tools.

Rules:
- Always use your tools for live travel information.
- Never invent flight availability, prices, airlines, dates, or times.
- Do not ask follow-up questions.
- If required information is missing, make the best possible tool-supported interpretation.
- Return concise structured information to the supervisor.
""",
    )

    return agent
