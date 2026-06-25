import os
import httpx
from openai import AzureOpenAI


def get_aoai_client():
    """Return a single AzureOpenAI client instance (created lazily)."""
    # Create client on demand to avoid import-time network calls
    client = AzureOpenAI(
        azure_endpoint="https://dev-aoai-api-all.jpn.mds.honda.com",
        api_key=os.environ.get("AZURE_OPENAI_API_KEY", ""),
        api_version="2025-03-01-preview",
        http_client=httpx.Client(
            verify=False,
            trust_env=False,
        ),
    )

    return client
