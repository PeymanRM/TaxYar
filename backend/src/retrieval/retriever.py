from dataclasses import dataclass
from typing import Any, Optional
from langchain_core.documents import Document
from langchain_qdrant import QdrantVectorStore
from retrieval.article_schema import RetrievedChunk

@dataclass
class RetrieverConfig:
    collection_name: str
    qdrant_url: str
    qdrant_api_key: str | None = None
    top_k: int = 5


class Retriever:
    def __init__(self, vector_store: QdrantVectorStore, top_k: int = 5):
        self.vector_store = vector_store
        self.top_k = top_k

    @classmethod
    def from_existing_collection(
        cls,
        *,
        embedding: Any,
        config: RetrieverConfig,
    ) -> "Retriever":
        vector_store = QdrantVectorStore.from_existing_collection(
            embedding=embedding,
            collection_name=config.collection_name,
            url=config.qdrant_url,
            # api_key=config.qdrant_api_key,
        )
        return cls(vector_store=vector_store, top_k=config.top_k)

    def retrieve(
        self,
        input: dict[str, str] | str,
        *,
        metadata_filter: Optional[dict[str, Any]] = None,
        top_k: Optional[int] = None,
    ) -> list[RetrievedChunk]:
        k = top_k or self.top_k

        if isinstance(input, dict):
            question = input.get("question", "")
        else:
            question = input

        docs_and_scores = self.vector_store.similarity_search_with_score(
            query=question,
            k=k,
            filter=metadata_filter,
        )
        results: list[RetrievedChunk] = []
        for doc, score in docs_and_scores:
            results.append(self._to_chunk(doc, score))

        return results

    def _to_chunk(self, doc: Document, score: float | None) -> RetrievedChunk:
        md = doc.metadata or {}

        return RetrievedChunk(
            text=doc.page_content,
            score=score,
            article_number=md.get("article_number"),
        )