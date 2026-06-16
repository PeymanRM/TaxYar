from src.indexing.qdrant_store import QdrantStore
import json
import tqdm
import argparse


parser = argparse.ArgumentParser(description='Index tax laws into Qdrant vector database.')
parser.add_argument('--embedded-file', default='../data/processed/embedded_direct_taxes_law_v1.json', help='Path to the JSON file containing tax law data.')
args = parser.parse_args()


if __name__ == '__main__':
    """Script to index tax laws into Qdrant vector database."""

    # 1. load data
    with open(args.embedded_file, "r") as f:
        data = json.load(f)

    # 2. init components
    store = QdrantStore()

    # 3. loop records
    with tqdm.tqdm(total=len(data), desc='Indexing laws') as pbar:
        for item in data:
            store.upsert(
                id=item["id"],
                vector=item["vector"],
                text=item["text"],
                metadata=item["metadata"]
            )
            pbar.update(1)