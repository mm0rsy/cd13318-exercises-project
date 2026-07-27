import os
import chromadb
from chromadb.config import Settings
from openai import OpenAI
from typing import Dict, List, Optional
from pathlib import Path

EMBEDDING_MODEL = "text-embedding-3-small"

def _get_query_embedding(query: str) -> List[float]:
    """Embed a query string using the same OpenAI model used during ingestion."""
    api_key = os.environ.get("OPENAI_API_KEY") or os.environ.get("CHROMA_OPENAI_API_KEY")
    client = OpenAI(api_key=api_key)
    response = client.embeddings.create(input=query, model=EMBEDDING_MODEL)
    return response.data[0].embedding

def discover_chroma_backends() -> Dict[str, Dict[str, str]]:
    """Discover available ChromaDB backends in the project directory"""
    backends = {}
    current_dir = Path(".")

    chroma_dirs = [
        d for d in current_dir.iterdir()
        if d.is_dir() and "chroma" in d.name.lower()
    ]

    for chroma_dir in chroma_dirs:
        try:
            db_client = chromadb.PersistentClient(
                path=str(chroma_dir),
                settings=Settings(anonymized_telemetry=False)
            )

            collections = db_client.list_collections()

            for collection in collections:
                key = f"{chroma_dir.name}::{collection.name}"

                try:
                    doc_count = collection.count()
                except Exception:
                    doc_count = 0

                backends[key] = {
                    "directory": str(chroma_dir),
                    "collection_name": collection.name,
                    "display_name": f"{chroma_dir.name} / {collection.name} ({doc_count} docs)",
                    "doc_count": str(doc_count),
                }

        except Exception as e:
            key = f"{chroma_dir.name}::error"
            error_msg = str(e)[:50]
            backends[key] = {
                "directory": str(chroma_dir),
                "collection_name": "",
                "display_name": f"{chroma_dir.name} (Error: {error_msg})",
                "doc_count": "0",
            }

    return backends

def initialize_rag_system(chroma_dir: str, collection_name: str):
    """Initialize the RAG system with specified backend (cached for performance)"""

    client = chromadb.PersistentClient(
        path=chroma_dir,
        settings=Settings(anonymized_telemetry=False)
    )
    collection = client.get_collection(collection_name)
    return collection, True, ""

def retrieve_documents(collection, query: str, n_results: int = 3, 
                      mission_filter: Optional[str] = None) -> Optional[Dict]:
    """Retrieve relevant documents from ChromaDB with optional filtering"""

    where_filter = None

    if mission_filter and mission_filter.lower() not in ("all", ""):
        where_filter = {"mission": {"$eq": mission_filter}}

    query_embedding = _get_query_embedding(query)

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=n_results,
        where=where_filter,
    )

    return results

def format_context(documents: List[str], metadatas: List[Dict]) -> str:
    """Format retrieved documents into context"""
    if not documents:
        return ""
    
    MAX_DOC_CHARS = 1500

    context_parts = ["=== Retrieved Context ==="]

    for i, (doc, meta) in enumerate(zip(documents, metadatas), start=1):
        mission = meta.get("mission", "unknown")
        mission = mission.replace("_", " ").title()

        source = meta.get("source", "unknown")

        category = meta.get("document_category", meta.get("data_type", "unknown"))
        category = category.replace("_", " ").title()

        header = f"\n[Source {i}] Mission: {mission} | File: {source} | Category: {category}"
        context_parts.append(header)

        content = doc[:MAX_DOC_CHARS] + "..." if len(doc) > MAX_DOC_CHARS else doc
        context_parts.append(content)

    return "\n".join(context_parts)