from langchain.agents import create_agent
from langchain_mcp_adapters.client import MultiServerMCPClient


WORLDCLOCK_MCP_URL = "https://worldclock.pro/mcp"


async def create_worldclock_agent(model):
    """
    Create the World Clock specialist agent.

    The agent owns all tools exposed by the World Clock MCP server.
    """

    mcp_client = MultiServerMCPClient(
        {
            "worldclock_server": {
                "transport": "streamable_http",
                "url": WORLDCLOCK_MCP_URL,
            }
        }
    )

    tools = await mcp_client.get_tools()

    agent = create_agent(
        model=model,
        tools=tools,
        system_prompt="""
You are the World Clock Specialist Agent.

Your responsibilities:
- Current time in cities.
- Time-zone conversion.
- Time by coordinates.
- DST information.
- City searches.

Rules:
- Always use your MCP tools for time information.
- Never guess the current time.
- Never invent time-zone information.
- Do not ask follow-up questions.
- Return concise information to the supervisor.
""",
    )

    return agent