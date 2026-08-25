"""Retrieval layer for SteelSense AI's RAG knowledge base.

Wraps the curated documents in knowledge_base.py in a Chroma vector
collection so the agent (added in a later stage) can ground its answers
in retrieved domain context instead of relying on the model's own
unverified priors.

The corpus is small (a handful of documents), so the collection is built
fresh in memory each time a KnowledgeBase is constructed rather than
persisted to disk -- simpler than managing an index file, and fast enough
that there's no real cost to rebuilding it on every process start.
"""

from __future__ import annotations

from dataclasses import dataclass

import chromadb

from .knowledge_base import KNOWLEDGE_BASE, KnowledgeDoc

_COLLECTION_NAME = "steelsense_knowledge"


@dataclass(frozen=True)
class RetrievedChunk:
    doc_id: str
    title: str
    category: str
    text: str
    source_note: str
    distance: float


class KnowledgeBase:
    def __init__(self, docs: list[KnowledgeDoc] | None = None) -> None:
        docs = docs if docs is not None else KNOWLEDGE_BASE
        self._client = chromadb.EphemeralClient()
        self._collection = self._client.create_collection(_COLLECTION_NAME)
        self._collection.add(
            ids=[doc.doc_id for doc in docs],
            documents=[doc.text for doc in docs],
            metadatas=[
                {"title": doc.title, "category": doc.category, "source_note": doc.source_note}
                for doc in docs
            ],
        )

    def retrieve(self, query: str, k: int = 3) -> list[RetrievedChunk]:
        """Return the k most relevant knowledge chunks for a query, ranked
        by embedding distance (lower is more relevant)."""
        result = self._collection.query(query_texts=[query], n_results=k)
        chunks = []
        for doc_id, text, meta, distance in zip(
            result["ids"][0],
            result["documents"][0],
            result["metadatas"][0],
            result["distances"][0],
        ):
            chunks.append(
                RetrievedChunk(
                    doc_id=doc_id,
                    title=meta["title"],
                    category=meta["category"],
                    text=text,
                    source_note=meta["source_note"],
                    distance=distance,
                )
            )
        return chunks


def _demo() -> None:
    """Manual sanity check: `python -m src.backend.rag.retrieval "<query>"`"""
    import sys

    query = " ".join(sys.argv[1:]) or "is cyclic loading a fatigue concern"
    kb = KnowledgeBase()
    print(f"Query: {query!r}\n")
    for chunk in kb.retrieve(query, k=3):
        print(f"[{chunk.distance:.3f}] {chunk.title} ({chunk.category})")
        print(f"    {chunk.text[:180]}...\n")


if __name__ == "__main__":
    _demo()
