import openai
from openai import AzureOpenAI
import os

import json
import httpx

from dotenv import load_dotenv

load_dotenv()
openai.api_type = "azure"

aoai_client = AzureOpenAI(
    azure_endpoint="https://dev-aoai-api-all.jpn.mds.honda.com",
    api_key=os.getenv("AZURE_OPENAI_API_KEY"), 
    api_version="2025-03-01-preview",
    http_client=httpx.Client(
        verify=False,
        trust_env=False,
        )
    )

if __name__ == "__main__":
    response = aoai_client.chat.completions.create(
      model="gpt-4.1",
      messages= 
      [
        {
          "role": "user",
          "content": [
            {"type": "text", "text": "この絵を説明してください。"},
            {
              "type": "image_url",
              "image_url": {
                "url": "https://www.petfamilyins.co.jp/pns/wp-content/uploads/2021/12/01_little-redhead-kitten-768x512.jpg.webp",
              },
            },
          ],
        }
      ],
      max_tokens=300
    )

    print(response.choices[0].message.content)