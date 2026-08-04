import os
from functools import lru_cache
from langchain_openai import ChatOpenAI
from langchain_cohere import ChatCohere

# Factory function to instantiate and cache the LLM instance (Singleton pattern)
@lru_cache(maxsize=1)
def get_llm(provider: str = None):
    provider = provider or os.getenv("LLM_PROVIDER", "openai").lower()

    if provider == "cohere":
        return ChatCohere(
            model=os.getenv("COHERE_MODEL", "command-r-08-2024"),
            temperature=0.2,
            max_tokens=1024,
        )
    elif provider == "openai":
        return ChatOpenAI(
            model=os.getenv("BASE_MODEL"),
            openai_api_base=os.getenv("BASE_MODEL_API_BASE"),
            openai_api_key=os.getenv("BASE_MODEL_API_KEY"),
        )
    else:
        raise ValueError(f"Unsupported LLM provider: {provider}")