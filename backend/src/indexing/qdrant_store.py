from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct

class QdrantStore:
    def __init__(self):
        self.client = QdrantClient(host="localhost", port=6333)
        self.collection = "tax-laws"

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