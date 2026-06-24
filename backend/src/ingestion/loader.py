import json
from pathlib import Path
from langchain_core.documents import Document

def load_laws(path):
    path = Path(path)

    with open(path, encoding="utf-8") as f:
        laws = json.load(f)

    documents = []

    for law in laws:
        text = law.get("text", "").strip()
        id = law.get("id", "").strip()

        if not text:
            continue

        metadata = {k: v for k, v in law.items() if k != "text" and k != "id"}

        doc = Document(
            id=id,
            page_content=text,
            metadata=metadata,
        )

        documents.append(doc)

    return documents

if __name__ == "__main__":
    path = Path(__file__).parent.parent.parent.parent / "data" / "raw" / "direct_taxes_law_v2.json"
    documents = load_laws(path)
    with open(Path(__file__).parent.parent.parent.parent / "data" / "raw" / "direct_taxes_law_v2_docs.json", "w", encoding="utf-8") as f:
        json.dump([doc.dict() for doc in documents], f, ensure_ascii=False, indent=4)