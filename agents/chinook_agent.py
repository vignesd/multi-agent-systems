from langchain.agents import create_agent
from langchain_mcp_adapters.client import MultiServerMCPClient
from middleware.agent_middleware import (
    get_summary_middleware,
    get_email_middleware,
    get_phone_number_middleware,
    get_prompt_injection_guard
)
from prompts.system_prompts import CUSTOMER_SYSTEM_PROMPT
import httpx
from dotenv import load_dotenv
import os

load_dotenv()


CHINOOK_MCP_SERVER = "https://chinook-mcp-server.onrender.com/mcp"
CHINOOK_MCP_SERVER_API_KEY = os.getenv("CHINOOK_MCP_SERVER_API_KEY", None)
if not CHINOOK_MCP_SERVER_API_KEY:
    raise ValueError("MCP Server API KEY is missing.")


async def create_customer_agent(model):

    mcp_client = MultiServerMCPClient(
        {
            "chinook": {
                "url": CHINOOK_MCP_SERVER,
                "transport": "streamable_http",
                "headers": {
                    "Authorization": f"Bearer {CHINOOK_MCP_SERVER_API_KEY}"
                },
            }

        }
    )

    try:
        tools = await mcp_client.get_tools()
        summary = get_summary_middleware(model)
        email = get_email_middleware()
        phone = get_phone_number_middleware()
        prompt_injection = get_prompt_injection_guard  # type:ignore

    except httpx.HTTPStatusError as e:
        if e.response.status_code == 401:
            print(
                "Error 401 Unauthorized: Check your API key, bearer token, or authentication credentials.")
        else:
            print(
                f"HTTP error occurred: {e.response.status_code} - {e.response.text}")

    except httpx.RequestError as e:
        print(f"An error occurred while making the request: {e}")

    customer_agent = create_agent(
        model=model,
        tools=tools,
        middleware=[
            summary,
            email,
            phone,
            prompt_injection,
        ],
        system_prompt=CUSTOMER_SYSTEM_PROMPT,
    )

    return customer_agent
