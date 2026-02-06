from rag import rag_answer
from web_search import serper_search
from groq import Groq
from dotenv import load_dotenv
import os

load_dotenv()

client = Groq(api_key=os.getenv("GROQ_API_KEY"))


def agent(query: str) -> str:
    needs_web = any(
        word in query.lower()
        for word in ["latest", "current", "news", "2024", "2025"]
    )

    if not needs_web:
        return rag_answer(query)

    web_context = serper_search(query)

    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[
            {
                "role": "system",
                "content": "Answer using the web information provided."
            },
            {
                "role": "user",
                "content": f"Web Info:\n{web_context}\n\nQuestion:\n{query}"
            }
        ],
        temperature=0
    )

    return response.choices[0].message.content
