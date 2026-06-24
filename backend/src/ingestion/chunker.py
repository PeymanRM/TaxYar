import json
import hashlib
from pathlib import Path
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

splitter = RecursiveCharacterTextSplitter(
    chunk_size=2000,
    chunk_overlap=400,
    separators=[
        "\n\n",
        "\n",
        ".",
        "،",
        " ",
        "\u200c",
        "",
    ],
)

def chunk_documents(documents: list[Document]) -> list[Document]:
    # Store source document ID in metadata before chunking
    for doc in documents:
        doc.metadata["article_id"] = doc.id
    
    chunked_docs = splitter.split_documents(documents)

    current_article_id = None
    chunk_index = 0    
    for doc in chunked_docs:
        article_id = doc.metadata.get("article_id")
        
        # Reset chunk index when source document changes
        if article_id != current_article_id:
            chunk_index = 0
            current_article_id = article_id
        
        doc.metadata["chunk_index"] = chunk_index
        new_id = article_id + "_" + str(chunk_index)
        doc.id = hashlib.md5(new_id.encode('utf-8')).hexdigest()
        chunk_index += 1
    
    return chunked_docs

if __name__ == "__main__":
    path = Path(__file__).parent.parent.parent.parent / "data" / "raw" / "direct_taxes_law_v2_docs.json"
    with open(path, "r", encoding="utf-8") as f:
        documents = [Document(**doc) for doc in json.load(f)]
    chunked_docs = chunk_documents(documents)
    with open(Path(__file__).parent.parent.parent.parent / "data" / "raw" / "direct_taxes_law_v2_chunks.json", "w", encoding="utf-8") as f:
        json.dump([doc.dict() for doc in chunked_docs], f, ensure_ascii=False, indent=4)