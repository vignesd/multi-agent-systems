import os
from dotenv import load_dotenv
from langchain.chat_models import BaseChatModel

load_dotenv()

MODEL = os.getenv("MODEL", "gpt-4o-mini")

# BASE_MODEL = BaseChatModel(
#     model=MODEL,
#     temperature=0.2,
# )