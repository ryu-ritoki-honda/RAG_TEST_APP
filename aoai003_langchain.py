from langchain_classic.chains import ConversationChain
from langchain_community.llms import AzureOpenAI
from langchain_classic.memory import ConversationBufferMemory
import os
import openai
import httpx
from dotenv import load_dotenv

load_dotenv()

openai.api_type = "azure"
openai.api_version = "2024-02-01"

chat = AzureOpenAI(
    model_name="gpt-4.1",
    openai_api_base="https://dev-aoai-api-all.jpn.mds.honda.com",
    openai_api_type="azure",
    openai_api_version="2025-03-01-preview",
    openai_api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    temperature=0.7,
    request_kwargs={"verify": False},
)

conversation = ConversationChain(
    llm=chat,
    memory=ConversationBufferMemory()
)

while True:
    user_message = input("You:")
    ai_message = conversation.run(input=user_message)
    print(f"AI: {ai_message}")
