import requests
import os
from dotenv import load_dotenv

load_dotenv()

class JinaEmbedder:
    def __init__(self):
        self.api_key = os.getenv("JINA_API_KEY")

    def embed(self, text: str):
        response = requests.post(
            "https://api.jina.ai/v1/embeddings",
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": "jina-embeddings-v4",
                "input": [text],
            },
        )

        response.raise_for_status()

        return response.json()["data"][0]["embedding"]