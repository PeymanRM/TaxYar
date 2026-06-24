# from src.indexing.embeddings import embedder


# def embed_laws(chunks, limit_embed_rate=100):
#     errors = []
#     embeddings = []
#     for i in range(0, len(chunks)//limit_embed_rate + 1):
#         print(f"Embedding laws {i*limit_embed_rate} to {(i+1)*limit_embed_rate}...")
#         with tqdm.tqdm(total=len(chunks[i*limit_embed_rate:(i+1)*limit_embed_rate]), desc='Embedding law chunks') as pbar:
#             for item in chunks[i*limit_embed_rate:(i+1)*limit_embed_rate]:
#                 try:
#                     vector = embedder.embed(item["page_content"])
#                     embeddings.append({
#                         "id": item["id"],
#                         "vector": vector
#                     })
#                 except Exception as e:
#                     print(f"Error embedding law {item['sectionId']}: {e}")
#                     errors.append(item["sectionId"])
#                     continue
                
#                 pbar.update(1)
#     print(f"Embedding completed with {len(errors)} errors. Error section IDs: {errors}")
#     return embeddings

"""
embed.py
--------
Step 1 of 2: Load chunks from JSON, embed their page_content, and save
the resulting vectors to disk so the indexing step can run independently.

Output file format (embeddings.json):
  [
    { "id": "<chunk_id>", "vector": [0.12, -0.03, ...] },
    ...
  ]
"""
import tqdm
import json
from pathlib import Path
from src.indexing.embeddings import embedder

def embed_chunks(
    chunks_path: str | Path,
    output_path: str | Path,
    batch_size: int = 32,
) -> None:
    chunks_path = Path(chunks_path)
    output_path = Path(output_path)

    # ── Load chunks ────────────────────────────────────────────────────────────
    print(f"Loading chunks from {chunks_path} ...")
    with open(chunks_path, encoding="utf-8") as f:
        raw_chunks = json.load(f)

    # Extract id and page_content; ignore 'type' and other LangChain internals
    ids   = [c["id"] for c in raw_chunks]
    texts = [c["page_content"] for c in raw_chunks]

    print(f"Embedding {len(texts)} chunks in batches of {batch_size} ...")

    # ── Embed in batches ───────────────────────────────────────────────────────
    all_vectors: list[list[float]] = []
    with tqdm.tqdm(total=len(texts), desc="Embedding law chunks") as pbar:
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            vectors = embedder.embed_documents(batch)
            all_vectors.extend(vectors)
            pbar.update(len(batch))

    # ── Save ───────────────────────────────────────────────────────────────────
    results = [{"id": id_, "vector": vec} for id_, vec in zip(ids, all_vectors)]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4, ensure_ascii=False)

    print(f"✓ Saved {len(results)} vectors to {output_path}")
    print(f"  Vector dimension: {len(all_vectors[0])}")


if __name__ == "__main__":
    embed_chunks(
        chunks_path="../data/raw/direct_taxes_law_v2_chunks.json",
        output_path="../data/processed/embeddings.json",
        batch_size=32,
    )