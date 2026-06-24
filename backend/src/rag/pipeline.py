from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from src.generation.prompts import ANSWER_PROMPT
from src.generation.formatters import format_chunks

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