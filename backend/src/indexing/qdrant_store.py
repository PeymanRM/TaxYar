from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct

class QdrantStore:
    def __init__(self):
        self.client = QdrantClient(host="localhost", port=6333)
        self.collection = "tax-laws"

    def create_collection(self, vector_size=2048, recreate=False):
        collections = [collection.name for collection in self.client.get_collections().collections]
        if self.collection not in collections:
            self.client.create_collection(
                collection_name=self.collection,
                vectors_config={
                    "size": vector_size,
                    "distance": "Cosine"
                }
            )
        elif recreate:
            self.client.delete_collection(collection_name=self.collection)
            self.client.create_collection(
                collection_name=self.collection,
                vectors_config={
                    "size": vector_size,
                    "distance": "Cosine"
                }
            )

    def upsert(self, id, vector, text, metadata):
        self.client.upsert(
            collection_name=self.collection,
            points=[
                PointStruct( #?
                    id=id,
                    vector=vector,
                    payload={
                        "text": text,
                        **metadata
                    }
                )
            ]
        )

    def search(self, query_vector, limit=5):
        return self.client.query_points(
            collection_name=self.collection,
            query=query_vector,
            limit=limit,
            with_payload=True,
        )