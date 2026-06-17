from src.indexing.embeddings import JinaEmbedder
from src.indexing.qdrant_store import QdrantStore
import json
import tqdm
import argparse


parser = argparse.ArgumentParser(description='Index tax laws into Qdrant vector database.')
parser.add_argument('--data-file', default='../data/raw/direct_taxes_law_v1.json', help='Path to the JSON file containing tax law data.')
parser.add_argument('--no-limit-embed', type=bool, default=True, help='Whether to limit the embedding rate to avoid hitting API rate limits.')
parser.add_argument('--limit-embed-rate', type=int, default=100, help='Limit the embedding rate to avoid hitting API rate limits.')
parser.add_argument('--recreate-collection', default=False, action='store_true', help='Whether to recreate the Qdrant collection before indexing.')
args = parser.parse_args()


if __name__ == '__main__':
    """Script to index tax laws into Qdrant vector database."""

    # 1. load data
    with open(args.data_file, 'r', encoding="utf-8") as f:
        data = json.load(f)

    # 2. init components
    embedder = JinaEmbedder()
    store = QdrantStore()
    store.create_collection(vector_size=2048, recreate=args.recreate_collection)

    # 3. loop records
    errors = []
    for i in range(0, len(data)//args.limit_embed_rate + 1):
        print(f"Indexing laws {i*args.limit_embed_rate} to {(i+1)*args.limit_embed_rate}...")
        with tqdm.tqdm(total=len(data[i*args.limit_embed_rate:(i+1)*args.limit_embed_rate]), desc='Indexing laws') as pbar:
            for item in data[i*args.limit_embed_rate:(i+1)*args.limit_embed_rate]:
                try:
                    vector = embedder.embed(item["finalSectionText"])
                except Exception as e:
                    print(f"Error embedding law {item['sectionId']}: {e}")
                    errors.append(item["sectionId"])
                    continue

                store.upsert(
                    id=item["sectionId"],
                    vector=vector,
                    text=item["finalSectionText"],
                    metadata={k: v for k, v in item.items() if k != "finalSectionText"}
                )
                pbar.update(1)
    print(f"Indexing completed with {len(errors)} errors. Error section IDs: {errors}")