from src.indexing.embeddings import get_embeddings
from src.indexing.qdrant_store import QdrantStore
import json
import tqdm
import argparse


parser = argparse.ArgumentParser(description='Index tax laws into Qdrant vector database.')
parser.add_argument('--data-file', default='data/raw/direct_taxes_law_v1.json', help='Path to the JSON file containing tax law data.')
parser.add_argument('--no-limit-embed', type=bool, default=True, help='Whether to limit the embedding rate to avoid hitting API rate limits.')
parser.add_argument('--limit-embed-rate', type=int, default=10, help='Limit the embedding rate to avoid hitting API rate limits.')
args = parser.parse_args()


if __name__ == '__main__':
    """Script to index tax laws into Qdrant vector database."""

    # 1. load data
    with open(args.data_file, "r") as f:
        data = json.load(f)

    # 2. init components
    embedder = get_embeddings()
    store = QdrantStore()

    # 3. loop records
    with tqdm.tqdm(total=len(data), desc='Indexing laws') as pbar:
        for item in data:
            vector = embedder.embed(item["finalSectionText"])

            store.upsert(
                id=item["sectionId"],
                vector=vector,
                text=item["finalSectionText"],
                metadata={k: v for k, v in item.items() if k != "finalSectionText"}
            )
            pbar.update(1)