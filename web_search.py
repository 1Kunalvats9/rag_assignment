import requests
import os
from dotenv import load_dotenv

load_dotenv()

SERPER_API_KEY = os.getenv("SERPER_API_KEY")

def serper_search(query: str) -> str:
    url = "https://google.serper.dev/search"

    headers = {
        "X-API-KEY": SERPER_API_KEY,
        "Content-Type": "application/json"
    }

    payload = {"q": query}

    response = requests.post(url, json=payload, headers=headers)
    results = response.json()

    snippets = [
        item.get("snippet", "")
        for item in results.get("organic", [])
    ]

    return "\n".join(snippets[:5])
