from tavily import TavilyClient



client=TavilyClient()

response=client.search("What is the top headlines today in INDIA",search_depth='basic',topic='news',max_results=1)


for i in response['results']:
    print(i['title'])
    print("="*50)
    print(i['content'])
    print(i['url'])
    print("\n")