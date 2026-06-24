import argparse
import json
from pathlib import Path
from qdrant_client.models import PointStruct
from src.indexing.qdrant_store import QdrantStore


def index_documents(
    chunks_path,
    embeddings_path,
    collection_name,
    vector_size = 1024,
    batch_size = 64,
    recreate = False,
):
    # Load chunks
    print(f"Loading chunks from {chunks_path} ...")
    with open(chunks_path, encoding="utf-8") as f:
        raw_chunks = json.load(f)
    chunks_by_id = {c["id"]: c for c in raw_chunks}

    # Load embeddings
    print(f"Loading embeddings from {embeddings_path} ...")
    with open(embeddings_path, encoding="utf-8") as f:
        embeddings = json.load(f)       # [{id, vector}, ...]
    print(f"{len(embeddings)} vectors loaded")

    # Prepare collection 
    QdrantStore.create_collection(
        collection_name=collection_name,
        vector_size=vector_size,
        recreate=recreate,
    )

    # Build points
    points = []
    missing = 0
    missing_ids = []

    for entry in embeddings:
        chunk_id = entry["id"]
        chunk = chunks_by_id.get(chunk_id)
        if chunk is None:
            missing += 1
            missing_ids.append(chunk_id)
            continue

        points.append(PointStruct(
            id=chunk_id,
            vector=entry["vector"],
            payload={
                "page_content": chunk["page_content"],
                "metadata": chunk.get("metadata", {}),
            },
        ))

    if missing:
        print(f"Warning: {missing} IDs had no matching chunk — skipped")
        print(f"Warning: Missing IDs: {missing_ids}")

    # Upsert in batches via QdrantStore 
    print(f"Upserting {len(points)} points into '{collection_name}' ...")
    for i in range(0, len(points), batch_size):
        batch = points[i : i + batch_size]
        QdrantStore.batch_upsert(collection_name=collection_name, points=batch)
        print(f"  {min(i + batch_size, len(points))}/{len(points)} upserted")

    print(f"{len(points)} documents indexed into '{collection_name}'")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Index documents into Qdrant")
    parser.add_argument("--chunks_path", type=str, default="../data/raw/direct_taxes_law_v2_chunks.json")
    parser.add_argument("--embeddings_path", type=str, default="../data/processed/direct_taxes_law_v2_jinav3_embeddings.json")
    parser.add_argument("--collection_name", type=str, default="direct-taxes-law")
    parser.add_argument("--vector_size", type=int, default=1024)
    parser.add_argument("--batch_size", type=int, default=64)
    args = parser.parse_args()

    index_documents(
        chunks_path=args.chunks_path,
        embeddings_path=args.embeddings_path,
        collection_name=args.collection_name,
        vector_size=1024,
        batch_size=64,
        recreate=True,
    )