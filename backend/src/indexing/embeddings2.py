from langchain_community.embeddings import JinaEmbeddings
from dotenv import load_dotenv

import os

load_dotenv()

embeddings = JinaEmbeddings(
    model_name="jina-embeddings-v4",
    jina_api_key=os.getenv("JINA_API_KEY")
)