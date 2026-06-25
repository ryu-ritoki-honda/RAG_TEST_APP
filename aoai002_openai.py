import os
import openai
from openai import AzureOpenAI
import json

import httpx

openai.api_type = "azure"
# print("HTTP_PROXY:", os.getenv("HTTP_PROXY"))
# print("HTTPS_PROXY:", os.getenv("HTTPS_PROXY"))
http_client = httpx.Client(
    trust_env=False,
    verify=False
)

aoai_client = AzureOpenAI(
    azure_endpoint="https://dev-aoai-api-all.jpn.mds.honda.com/",
    api_key=os.environ["AZURE_OPENAI_API_KEY"],
    api_version="2025-03-01-preview",
    http_client=http_client
)

if __name__ == "__main__":
    response = aoai_client.chat.completions.create(
      model="gpt-4.1",
      messages = 
      [
        {
          "role": "system",
          "content": "You are an AI assistant that helps people find information."
        },
        {
          "role": "user",
          "content": "犬は猫ですか？"
        },
        {
          "role": "assistant",
          "content": "いいえ、犬は猫ではありません。犬は犬科の動物であり、猫はネコ科の動物です。両方の動物はペットとして人気がありますが、種類や性格が異なるため、異なる世話や注意が必要です。"
        },
        {
          "role": "user",
          "content": "百田夏菜子とデートしたい。"
        }
      ],
      temperature=0.5,
      max_tokens=800,
      top_p=0.95,
      frequency_penalty=0,
      presence_penalty=0,
      stop=None)

    print(response)