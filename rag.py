from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from web_search import serper_search
from groq import Groq
from dotenv import load_dotenv
import os

load_dotenv()

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DATA_PATH = "data"
INDEX_PATH = "faiss_index"


def _load_documents_from_directory(directory: str) -> list[Document]:
    documents: list[Document] = []
    if not os.path.isdir(directory):
        return documents

    for name in os.listdir(directory):
        path = os.path.join(directory, name)
        if not os.path.isfile(path):
            continue
        # For now we only index plain text files.
        if not name.lower().endswith(".txt"):
            continue
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            content = f.read().strip()
        if not content:
            continue
        documents.append(
            Document(page_content=content, metadata={"source": path})
        )
    return documents


def create_vector_store():
    documents = _load_documents_from_directory(DATA_PATH)

    if not documents:
        raise ValueError("No .txt documents found in data directory.")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )
    chunks = splitter.split_documents(documents)

    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

    vectorstore = FAISS.from_documents(chunks, embeddings)
    vectorstore.save_local(INDEX_PATH)

    print("✅ FAISS index created")
    return vectorstore


def load_vector_store():
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    return FAISS.load_local(
        INDEX_PATH,
        embeddings,
        allow_dangerous_deserialization=True
    )


def rag_answer(query: str) -> str:
    db = load_vector_store()
    docs = db.similarity_search(query, k=3)

    context = "\n".join(d.page_content for d in docs)

    # Simple confidence heuristic: very short or empty context means low RAG confidence.
    low_confidence = not context or len(context) < 200

    web_context = ""
    if low_confidence:
        # Fall back to Serper-powered web search when local context is weak.
        web_context = serper_search(query)

    if web_context:
        combined_context = f"Local context:\n{context}\n\nWeb search context:\n{web_context}"
    else:
        combined_context = f"Local context:\n{context}"

    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[
            {
                "role": "system",
                "content": (
                    "Use the provided context to answer the question. "
                    "Prefer local context when it clearly answers the question, "
                    "but you may also use web search context when local context is missing or insufficient."
                ),
            },
            {
                "role": "user",
                "content": f"Context:\n{combined_context}\n\nQuestion:\n{query}",
            },
        ],
        temperature=0,
    )

    return response.choices[0].message.content
