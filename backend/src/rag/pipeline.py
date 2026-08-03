import os
from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from generation.prompts import ANSWER_PROMPT
from generation.formatters import format_chunks

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from retrieval.retriever import Retriever, RetrieverConfig
from indexing.embeddings import embedder
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
# llm = ChatCohere(
#         model="command-r-08-2024",   # or "command-r"
#         temperature=0.2,
#         max_tokens=1024,
#     )
# !!!
llm = ChatOpenAI(
    model=os.getenv("BASE_MODEL"),
    openai_api_base=os.getenv("BASE_MODEL_API_BASE"),
    openai_api_key=os.getenv("BASE_MODEL_API_KEY"),
)
retriever = Retriever.from_existing_collection(
    embedding=embedder,
    config=RetrieverConfig(
        collection_name=os.getenv("DIRECT_TAX_LAW_COLLECTION_NAME"),
        qdrant_url=os.getenv("VECTOR_DB_URL"),
        # qdrant_api_key=None,
        top_k=int(os.getenv("SIMILARITY_SEARCH_TOP_K", 5)),
    )
)

rag_chain = build_rag_chain(retriever, llm)