from dataclasses import dataclass
from typing import Any, Optional
from functools import lru_cache
import os
from langchain_core.callbacks import StdOutCallbackHandler

from langchain_core.documents import Document
from langchain_qdrant import QdrantVectorStore

# As of langchain 1.x, SelfQueryRetriever (and the query_constructor/chains
# machinery it depends on) was split out of core `langchain` into the
# separate `langchain_classic` package. Install it with:
#   pip install -U langchain-classic
from langchain_classic.retrievers.self_query.base import SelfQueryRetriever

# NOTE: import the translator directly instead of letting SelfQueryRetriever
# auto-detect it. Auto-detection (_get_builtin_translator) eagerly imports
# every vectorstore class in langchain_community.vectorstores -- including
# DatabricksVectorSearch -- which is what triggers the ImportError you saw.
# It also only recognizes the legacy `Qdrant` class, not `QdrantVectorStore`
# from langchain_qdrant, so auto-detection would fail for your setup anyway.
from langchain_community.query_constructors.qdrant import QdrantTranslator

from retrieval.article_schema import (
    RetrievedChunk,
    metadata_field_info,
    document_content_description,
)
from indexing.embeddings import embedder
from generation.llm import get_llm


@dataclass
class SelfQueryRetrieverConfig:
    collection_name: str
    qdrant_url: str
    qdrant_api_key: str | None = None
    top_k: int = 5
    verbose: bool = True


class SelfQueryingRetriever:
    def __init__(self, self_query_retriever: SelfQueryRetriever):
        self._retriever = self_query_retriever

    @classmethod
    def from_existing_collection(
        cls,
        *,
        embedding: Any,
        config: SelfQueryRetrieverConfig,
        llm: Any = None,
    ) -> "SelfQueryingRetriever":
        vector_store = QdrantVectorStore.from_existing_collection(
            embedding=embedding,
            collection_name=config.collection_name,
            url=config.qdrant_url,
            # api_key=config.qdrant_api_key,
        )

        # QdrantVectorStore stores metadata under this payload key.
        # Defaults to "metadata" but read it off the instance to be safe
        # across langchain-qdrant versions.
        metadata_key = getattr(vector_store, "metadata_payload_key", "metadata")

        self_query_retriever = SelfQueryRetriever.from_llm(
            llm=llm or get_llm(),
            vectorstore=vector_store,
            document_contents=document_content_description,
            metadata_field_info=metadata_field_info,
            structured_query_translator=QdrantTranslator(metadata_key=metadata_key),
            search_kwargs={"k": config.top_k},
            verbose=config.verbose,
            use_original_query=True, 
        )

        return cls(self_query_retriever=self_query_retriever)

    def retrieve(
        self,
        input: dict[str, str] | str,
    ) -> list[RetrievedChunk]:
        if isinstance(input, dict):
            question = input.get("question", "")
        else:
            question = input

        docs: list[Document] = self._retriever.invoke(question, config={"callbacks": [StdOutCallbackHandler()]})

        # SelfQueryRetriever doesn't return similarity scores (the LLM-built
        # structured filter + similarity_search under the hood discards them
        # at this abstraction level), so score is None here.
        return [self._to_chunk(doc) for doc in docs]

    def _to_chunk(self, doc: Document) -> RetrievedChunk:
        md = doc.metadata or {}
        return RetrievedChunk(
            text=doc.page_content,
            score=None,
            article_number=md.get("article_number"),
        )


@lru_cache(maxsize=1)
def get_self_query_retriever() -> SelfQueryingRetriever:
    config = SelfQueryRetrieverConfig(
        collection_name=os.getenv("DIRECT_TAX_LAW_COLLECTION_NAME", ""),
        qdrant_url=os.getenv("VECTOR_DB_URL", ""),
        qdrant_api_key=os.getenv("VECTOR_DB_API_KEY"),
        top_k=int(os.getenv("SIMILARITY_SEARCH_TOP_K", "5")),
        verbose=True,
    )
    return SelfQueryingRetriever.from_existing_collection(
        embedding=embedder,
        config=config,
    )