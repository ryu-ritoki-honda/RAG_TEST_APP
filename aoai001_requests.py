import requests
import os
from dotenv import load_dotenv

load_dotenv()

url = "https://dev-aoai-api-all.jpn.mds.honda.com/openai/deployments/gpt-4.1/chat/completions"
querystring = {"api-version":"2025-03-01-preview"}

payload = {"messages": [
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
    ]}
headers = {
    "Content-Type": "application/json",
    "api-key": os.getenv("AZURE_OPENAI_API_KEY"),
    "User-Agent": ""
}
path = os.getcwd()
response = requests.request("POST", url, json=payload, headers=headers, params=querystring, verify=False)

print(response.text)