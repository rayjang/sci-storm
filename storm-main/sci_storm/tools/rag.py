from __future__ import annotations

from pathlib import Path
from typing import Iterable, List


class LocalRAGClient:
    """
    Minimal placeholder for a local vector index.

    The class exposes a uniform `ingest` and `query` API so that we can swap the
    implementation between ChromaDB or FAISS during later iterations without
    touching the pipeline logic.
    """

    def __init__(self, persist_directory: Path):
        self.persist_directory = Path(persist_directory)
        self.persist_directory.mkdir(parents=True, exist_ok=True)
        self.documents: List[str] = []

    def ingest(self, docs: Iterable[str]):
        for doc in docs:
            self.documents.append(doc)

    def query(self, query: str, k: int = 5) -> List[str]:
        # Placeholder: future iteration will route to an embedding model and vector store
        if not self.documents:
            return []
        return self.documents[:k]

