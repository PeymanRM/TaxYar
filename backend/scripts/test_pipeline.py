from dotenv import load_dotenv
from src.retrieval.retriever import Retriever, RetrieverConfig
from src.indexing.embeddings import embedder
from src.rag.pipeline import build_rag_chain
from langchain_cohere import ChatCohere
# from langchain_openrouter import ChatOpenRouter
from bidi.algorithm import get_display
import arabic_reshaper
import os

load_dotenv()

def terminal_text(text: str) -> str:
    # Keeps Persian connected and reorders RTL/LTR for terminal display
    return get_display(arabic_reshaper.reshape(text))

load_dotenv()

if __name__=="__main__":
    # llm = ChatOpenRouter(
    #     model="meta-llama/llama-3.3-70b-instruct:free", 
    #     max_tokens=1024,
    #     max_retries=2,
    # )
    llm = ChatCohere(
        model="command-r-08-2024",   # or "command-r"
        temperature=0.2,
        max_tokens=1024,
    )
    retriever = Retriever.from_existing_collection(
        embedding=embedder,
        config = RetrieverConfig(
            collection_name=os.getenv("DIRECT_TAX_LAW_COLLECTION_NAME", ""),
            qdrant_url=os.getenv("VECTOR_DB_URL", ""),
            qdrant_api_key=os.getenv("VECTOR_DB_API_KEY"),
            top_k=int(os.getenv("SIMILARITY_SEARCH_TOP_K", "5")),
        )
    )

    chain = build_rag_chain(retriever, llm)

    while True:
        question = input("\nEnter your question (or 'q' to quit): ")
        if question.lower() == "q":
            break
        response =chain.invoke(question)
        for i, chunk in enumerate(response["context"], start=1):
            print(f"\n=== Retrieved Chunk {i} ===")
            print("Article Number:", chunk.article_number)
            print("Text:", terminal_text(chunk.text))
            print("Score:", chunk.score)
        print("\n=== Final Answer ===")
        print("Text:", terminal_text(response["answer"]))