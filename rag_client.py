import os
import chromadb
from chromadb.config import Settings
from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction
from typing import Dict, List, Optional
from pathlib import Path

# Must match the default embedding model used by embedding_pipeline.py so
# that queries are embedded in the same vector space as the stored documents.
DEFAULT_EMBEDDING_MODEL = "text-embedding-3-small"


def _resolve_base_url() -> Optional[str]:
    """Resolve a custom OpenAI-compatible base URL (e.g. Vocareum proxy).

    Checks both the current OPENAI_BASE_URL and the older, still commonly
    documented OPENAI_API_BASE, since the openai SDK only auto-reads the
    former.
    """
    return os.getenv("OPENAI_BASE_URL") or os.getenv("OPENAI_API_BASE")


def discover_chroma_backends() -> Dict[str, Dict[str, str]]:
    """Discover available ChromaDB backends in the project directory"""
    backends = {}
    current_dir = Path(".")

    # Look for ChromaDB directories: any top-level directory whose name
    # contains "chroma" (e.g. chroma_db, chroma_db_openai) - these are the
    # persist directories created by embedding_pipeline.py.
    candidate_dirs = [
        d for d in current_dir.iterdir()
        if d.is_dir() and "chroma" in d.name.lower()
    ]

    for chroma_dir in candidate_dirs:
        try:
            # Initialize database client with directory path and configuration settings
            client = chromadb.PersistentClient(
                path=str(chroma_dir),
                settings=Settings(anonymized_telemetry=False)
            )

            # Retrieve list of available collections from the database
            collections = client.list_collections()

            for collection in collections:
                collection_name = collection.name

                # Create unique identifier key combining directory and collection names
                key = f"{chroma_dir.name}::{collection_name}"

                # Get document count with fallback for unsupported operations
                try:
                    count = client.get_collection(collection_name).count()
                except Exception:
                    count = "unknown"

                # Build information dictionary
                backends[key] = {
                    "directory": str(chroma_dir),
                    "collection_name": collection_name,
                    "display_name": f"{chroma_dir.name} / {collection_name} ({count} docs)",
                    "document_count": str(count),
                }

        except Exception as e:
            # Handle connection or access errors gracefully - create a
            # fallback entry so the user can still see the directory exists
            # even if it couldn't be opened as a valid ChromaDB store.
            key = f"{chroma_dir.name}::error"
            backends[key] = {
                "directory": str(chroma_dir),
                "collection_name": "",
                "display_name": f"{chroma_dir.name} (unavailable: {str(e)[:50]})",
                "document_count": "0",
            }

    return backends


def initialize_rag_system(chroma_dir: str, collection_name: str,
                           embedding_model: str = DEFAULT_EMBEDDING_MODEL):
    """Initialize the RAG system with specified backend (cached for performance)

    Returns a (collection, success, error) tuple so callers can distinguish
    a working connection from a failed one without relying on exceptions.
    """
    try:
        # The embedding function must match the one used at ingestion time
        # (embedding_pipeline.py) so queries land in the same vector space.
        # It reads the API key from the CHROMA_OPENAI_API_KEY (or
        # OPENAI_API_KEY) environment variable, and base_url is resolved
        # explicitly to support classroom proxy keys (e.g. Vocareum).
        embedding_function = OpenAIEmbeddingFunction(
            model_name=embedding_model,
            api_base=_resolve_base_url()
        )

        # Create a chromadb persistent client
        client = chromadb.PersistentClient(
            path=chroma_dir,
            settings=Settings(anonymized_telemetry=False)
        )

        # Return the collection with the collection_name
        collection = client.get_collection(
            name=collection_name,
            embedding_function=embedding_function
        )
        return collection, True, None
    except Exception as e:
        return None, False, str(e)


def retrieve_documents(collection, query: str, n_results: int = 3,
                      mission_filter: Optional[str] = None) -> Optional[Dict]:
    """Retrieve relevant documents from ChromaDB with optional filtering"""

    # Initialize filter variable to None (represents no filtering)
    where_filter = None

    # Check if filter parameter exists and is not set to "all"/equivalent
    if mission_filter and mission_filter.strip().lower() not in ("all", "none", ""):
        # Create filter dictionary with appropriate field-value pairs
        where_filter = {"mission": mission_filter}

    # Execute database query
    results = collection.query(
        query_texts=[query],
        n_results=n_results,
        where=where_filter
    )

    # Return query results to caller
    return results


def format_context(documents: List[str], metadatas: List[Dict]) -> str:
    """Format retrieved documents into context"""
    if not documents:
        return ""

    # Initialize list with header text for context section
    context_parts = ["Retrieved NASA Mission Documents:"]

    seen_snippets = set()
    max_doc_length = 1200

    # Loop through paired documents and their metadata using enumeration
    for i, (doc, metadata) in enumerate(zip(documents, metadatas), start=1):
        metadata = metadata or {}

        doc_text = (doc or "").strip()
        if not doc_text:
            continue

        # Deduplicate near-identical snippets (skip repeats, don't renumber)
        fingerprint = doc_text[:200]
        if fingerprint in seen_snippets:
            continue
        seen_snippets.add(fingerprint)

        # Extract mission information from metadata with fallback value
        mission = metadata.get("mission", "unknown_mission")
        # Clean up mission name formatting (replace underscores, capitalize)
        mission = mission.replace("_", " ").title()

        # Extract source information from metadata with fallback value
        source = metadata.get("source") or metadata.get("file_path", "unknown_source")

        # Extract category information from metadata with fallback value
        category = metadata.get("document_category", "general_document")
        # Clean up category name formatting (replace underscores, capitalize)
        category = category.replace("_", " ").title()

        # Create formatted source header with index number and extracted information
        header = f"\n[Source {i} | Mission: {mission} | Document: {source} | Category: {category}]"
        # Add source header to context parts list
        context_parts.append(header)

        # Check document length and truncate if necessary
        if len(doc_text) > max_doc_length:
            doc_text = doc_text[:max_doc_length].rstrip() + " ... [truncated]"
        # Add truncated or full document content to context parts list
        context_parts.append(doc_text)

    # Join all context parts with newlines and return formatted string
    return "\n".join(context_parts)