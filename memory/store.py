"""
store.py — the long-term memory layer.

This wraps a persistent Chroma vector store. Every "memory" is a short
piece of text (a fact, a summary, or a raw exchange) plus metadata
(timestamp, type, importance). Semantic search (embedding similarity)
is how the chatbot decides what's "relevant" to recall for a given query.
"""

import time
import uuid
from typing import List, Dict, Optional

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

import config


class MemoryStore:
    def __init__(self):
        # Local embedding model — no API calls, runs on CPU
        self.embeddings = HuggingFaceEmbeddings(model_name=config.EMBEDDING_MODEL)

        self.vectorstore = Chroma(
            collection_name=config.MEMORY_COLLECTION_NAME,
            embedding_function=self.embeddings,
            persist_directory=config.CHROMA_PERSIST_DIR,
        )

    def add_memory(
        self,
        text: str,
        memory_type: str = "fact",
        importance: float = 0.5,
        extra_metadata: Optional[Dict] = None,
    ) -> str:
        """Store a single memory. Returns the memory's id."""
        memory_id = str(uuid.uuid4())
        metadata = {
            "type": memory_type,          # "fact" | "summary" | "raw_turn"
            "importance": importance,     # 0-1, used later for pruning/ranking
            "timestamp": time.time(),
            "id": memory_id,
        }
        if extra_metadata:
            metadata.update(extra_metadata)

        self.vectorstore.add_texts(texts=[text], metadatas=[metadata], ids=[memory_id])
        return memory_id

    def retrieve_relevant(self, query: str, k: int = None) -> List[Dict]:
        """Semantic search over clean extracted facts only (excludes raw_turn
        logs, which are kept purely for debugging, not for recall context)."""
        k = k or config.RETRIEVAL_TOP_K
        results = self.vectorstore.similarity_search_with_score(
            query, k=k, filter={"type": "fact"}
        )

        memories = []
        for doc, score in results:
            memories.append(
                {
                    "text": doc.page_content,
                    "metadata": doc.metadata,
                    "similarity_score": score,
                }
            )
        return memories

    def get_all_memories(self, memory_type: Optional[str] = None) -> List[Dict]:
        """Fetch every stored memory, optionally filtered by type.
        Used by the consolidator to decide what to compress."""
        raw = self.vectorstore.get(include=["documents", "metadatas"])
        memories = []
        for doc, meta, doc_id in zip(raw["documents"], raw["metadatas"], raw["ids"]):
            if memory_type is None or meta.get("type") == memory_type:
                memories.append({"id": doc_id, "text": doc, "metadata": meta})
        return memories

    def delete_memories(self, ids: List[str]):
        if ids:
            self.vectorstore.delete(ids=ids)

    def count(self, memory_type: Optional[str] = None) -> int:
        return len(self.get_all_memories(memory_type))
