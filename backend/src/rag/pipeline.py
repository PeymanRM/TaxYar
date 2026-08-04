from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from generation.prompts import ANSWER_PROMPT
from generation.formatters import format_chunks
from generation.llm import get_llm
from retrieval.retriever import get_retriever
from functools import lru_cache
from dotenv import load_dotenv

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

    
@lru_cache(maxsize=1)
def get_rag_chain():
    """Singleton factory to get assembled pipeline."""
    return build_rag_chain(
        retriever=get_retriever(),
        llm=get_llm(provider="cohere"),
    )