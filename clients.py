from clients.chinook_client import run_customer_chat
from clients.travel_client import run_travel_chat
import asyncio


async def run_agent():
    await run_customer_chat()


if __name__ == "__main__":
    asyncio.run(run_agent())
