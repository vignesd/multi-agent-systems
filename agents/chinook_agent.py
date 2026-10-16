from langchain.agents import create_agent
from langchain_mcp_adapters.client import MultiServerMCPClient
from middleware.agent_middleware import (
    get_default_middleware,
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


async def create_customer_agent(model):
    api_key = os.getenv("CHINOOK_MCP_SERVER_API_KEY")
    if not api_key:
        raise ValueError("CHINOOK_MCP_SERVER_API_KEY is missing.")

    mcp_client = MultiServerMCPClient(
        {
            "chinook": {
                "url": CHINOOK_MCP_SERVER,
                "transport": "streamable_http",
                "headers": {
                    "Authorization": f"Bearer {api_key}"
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
        raise

    except httpx.RequestError as e:
        print(f"An error occurred while making the request: {e}")
        raise

    customer_agent = create_agent(
        model=model,
        tools=tools,
        middleware=[
            # Runs first so a rejected request ends before anything else.
            prompt_injection,
            *get_default_middleware(model),
            summary,
            email,
            phone,
        ],
        system_prompt=CUSTOMER_SYSTEM_PROMPT,
    )

    return customer_agent
