import inspect
from langchain_openai import AzureChatOpenAI
print('signature:', inspect.signature(AzureChatOpenAI.__init__))
print('module:', AzureChatOpenAI.__module__)
print('source snippet:')
print(inspect.getsource(AzureChatOpenAI)[:2000])
