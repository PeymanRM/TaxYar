from src.retrieval.retriever import Retriever, RetrieverConfig
from src.indexing.embeddings import embedder
from dotenv import load_dotenv
from bidi.algorithm import get_display
import arabic_reshaper
import unicodedata
import json
import os

load_dotenv()

def clean_text(text: str) -> str:
    text = unicodedata.normalize("NFC", text)
    return (
        text.replace("<br>", "\n")
            .replace("&nbsp;", " ")
            .strip()
    )

def terminal_text(text: str) -> str:
    # Keeps Persian connected and reorders RTL/LTR for terminal display
    return get_display(arabic_reshaper.reshape(text))

def main() -> None:
    with open("../data/eval/examples1.json", encoding="utf-8") as f:
        examples = json.load(f)

    question = examples[5]["question"]

    retriever = Retriever.from_existing_collection(
        embedding=embedder,
        config=RetrieverConfig(
            collection_name="direct-taxes-law",
            qdrant_url="http://localhost:6333",
            # qdrant_api_key=None,
            top_k=5,
        ),
    )

    results = retriever.retrieve(question)

    print(f"Question: {terminal_text(clean_text(question))}")

    for i, chunk in enumerate(results, start=1):
        print(f"\n=== Result {i} ===")
        print("Text:", terminal_text(clean_text(chunk.text)))
        print("Score:", chunk.score)
        print("Metadata:", chunk.metadata)
        # print("Article:", chunk.article)
        # print("Source:", chunk.source)
        # print("Page:", chunk.page)
        # print("ChunkId:", chunk.chunk_id)


if __name__ == "__main__":
    main()