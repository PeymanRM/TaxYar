from retrieval.article_schema import RetrievedChunk

def format_chunks(chunks: list[RetrievedChunk]) -> str:
    blocks = []

    for chunk in chunks:
        blocks.append(
            f"[Article number: {chunk.article_number}]\n{chunk.text}"
        )

    return "\n\n---\n\n".join(blocks)

def get_ranks(chunks: list[RetrievedChunk]) -> list[int]:
    return [chunk.article_number for chunk in chunks]