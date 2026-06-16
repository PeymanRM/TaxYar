from langchain.embeddings import HuggingFaceEmbeddings

def get_embeddings():
    return HuggingFaceEmbeddings(
        model_name="jinaai/jina-embeddings-v4"
    )