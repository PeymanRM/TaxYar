import os
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams


class QdrantStore:

    client = QdrantClient(
        url=os.getenv("QDRANT_URL", "http://localhost:6333"),
        api_key=os.getenv("QDRANT_API_KEY"),   # None → no auth (local)
    )

    @classmethod
    def create_collection(
        cls,
        collection_name: str,
        vector_size: int = 1024,
        recreate: bool = False,
    ) -> None:
        exists = collection_name in [
            c.name for c in cls.client.get_collections().collections
        ]

        if exists and recreate:
            cls.client.delete_collection(collection_name)
            exists = False

        if not exists:
            cls.client.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(
                    size=vector_size,
                    distance=Distance.COSINE,
                ),
            )

    @classmethod
    def batch_upsert(
        cls,
        collection_name: str,
        points: list[PointStruct],
    ) -> None:
        cls.client.upsert(
            collection_name=collection_name,
            points=points,
        )

    @classmethod
    def upsert(cls, collection_name, id, vector, text, metadata):
        cls.client.upsert(
            collection_name=collection_name,
            points=[
                PointStruct(
                    id=id,
                    vector=vector,
                    payload={
                        "text": text,
                        **metadata,
                    },
                )
            ],
        )

    @classmethod
    def search(cls, collection_name, query_vector, limit=5):
        return cls.client.query_points(
            collection_name=collection_name,
            query=query_vector,
            limit=limit,
            with_payload=True,
        )