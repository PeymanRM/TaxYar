from src.retrieval.article_schema import RetrievedChunk

def format_chunks(chunks: list[RetrievedChunk]) -> str:
    blocks = []

    for chunk in chunks:
        blocks.append(
            f"[Article number: {chunk.article_number}]\n{chunk.text}"
        )

    return "\n\n---\n\n".join(blocks)