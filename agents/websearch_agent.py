# from tavily import TavilyClient
from dotenv import load_dotenv
import os
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain.agents import create_agent
import asyncio


load_dotenv()
api_key=os.getenv("TAVILY_API_KEY")
WEBSEARCH_MCP_URL=f"https://mcp.tavily.com/mcp/?tavilyApiKey={api_key}"

async def create_websearch_agent(model):

    mcp_client = MultiServerMCPClient(
        {
            "worldclock_server": {
                "transport": "streamable_http",
                "url": WEBSEARCH_MCP_URL,
            }
        }
    )

    tools =await mcp_client.get_tools()
    
    agent=create_agent(
        model=model,
        tools=tools,
        system_prompt="""
You are the Web Research Specialist Agent.

Your responsibilities:

* Web searches for real-time news, facts, and current information (`tavily_search`).
* Full-page content extraction from specific URLs (`tavily_extract`).
* Website crawling across subpages with configurable depth (`tavily_crawl`).
* Mapping site architecture and discovering page URL structures (`tavily_map`).
* Comprehensive multi-source deep-dive topic research (`tavily_research`).

Rules:

* Always use your Tavily tools for web information beyond static training data.
* Never guess web content, site structures, or real-time facts.
* Rely strictly on the 5 available Tavily tools; never invent tool capabilities.
* Mind the 20 requests/minute rate limit on `tavily_research`.
* Do not ask follow-up questions.
* Return concise, accurate, and properly cited information to the supervisor.
"""
    )

    return agent
    


# if __name__=="__main__":
#     asyncio.run(create_websearch_agent())

















































# client=TavilyClient()
# # response=client.search("What is the top headlines today in INDIA",search_depth='basic',topic='news',max_results=1)

# # for i in response['results']:
# #     print(i['title'])
# #     print("="*50)
# #     print(i['content'])
# #     print(i['url'])
# #     print("\n")

# print(client.__doc__)