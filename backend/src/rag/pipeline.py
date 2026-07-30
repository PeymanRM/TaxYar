from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from src.generation.prompts import ANSWER_PROMPT
from src.generation.formatters import format_chunks

from dotenv import load_dotenv
from langchain_cohere import ChatCohere
from src.retrieval.retriever import Retriever, RetrieverConfig
from src.indexing.embeddings import embedder
load_dotenv()

def build_rag_chain(retriever, llm):
    rag_chain = (
        RunnableParallel(
            {
                "context": RunnableLambda(retriever.retrieve),
                "question": RunnablePassthrough(),
            }
        )
        .assign(
            answer=(
                RunnablePassthrough.assign(
                    context=lambda x: format_chunks(x["context"])
                )
                | ANSWER_PROMPT
                | llm
                | StrOutputParser()
            )
        )
    )

    return rag_chain

# TODO SEPERATE THIS AS A MODULE or sth
llm = ChatCohere(
        model="command-r-08-2024",   # or "command-r"
        temperature=0.2,
        max_tokens=1024,
    )
retriever = Retriever.from_existing_collection(
    embedding=embedder,
    config=RetrieverConfig(
        collection_name="direct-taxes-law",
        qdrant_url="http://localhost:6333",
        # qdrant_api_key=None,
        top_k=3,
    )
)

rag_chain = build_rag_chain(retriever, llm)