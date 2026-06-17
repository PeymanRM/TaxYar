from src.indexing.embeddings import JinaEmbedder
from src.indexing.qdrant_store import QdrantStore
import json

examples=[]

embedder = JinaEmbedder()
store = QdrantStore()

with open('../data/eval/examples1.json', encoding='utf-8') as f:
    examples = json.load(f)

query_vector = embedder.embed(examples[5]["question"])

results = store.search(query_vector, limit=5)
# print(f"Top {len(results.points)} results for query: '{query}'")
# print("=" * 50)
# print("Results:")
# print(results)
# print("=" * 50)

print(f"Question: {examples[5]['question']}")

for i, point in enumerate(results.points, start=1):
    print(f"\n=== Result {i} ===")
    print("point:", point)
    print("Score:", point.score)
    print("SectionId:", point.id)
    # print("Text:", point.payload["text"][:9])