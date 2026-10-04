from langchain.chat_models import BaseChatModel

from langchain.agents import create_agent
from langchain.messages import HumanMessage, SystemMessage
from langchain_mcp_adapters.client import MultiServerMCPClient


from dotenv import load_dotenv
import os
load_dotenv()

MODEL = os.getenv("MODEL", "gpt-4o-mini")
BASE_MODEL = BaseChatModel(
    model=MODEL,
    temperature=0.2,
)


TRAVEL_MCP_URL = "https://mcp.kiwi.com"
WORLDCLOCK_MCP_URL = "https://worldclock.pro/mcp"
CURRENCY_MCP_URL = "https://currency-mcp.wesbos.com/mcp"


async def create_travel_agent():
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
        model=BASE_MODEL,
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


async def create_currency_agent():
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
        model=BASE_MODEL,
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


async def create_worldclock_agent():
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
        model=BASE_MODEL,
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
