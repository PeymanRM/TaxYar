from operator import itemgetter
from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, trim_messages

from generation.prompts import ANSWER_PROMPT
from generation.formatters import format_chunks
from generation.llm import get_llm
from retrieval.retriever import get_retriever
from functools import lru_cache
from dotenv import load_dotenv

load_dotenv()


def convert_frontend_messages(history):
    """Converts history items (dict, Pydantic model, or LangChain Message) into standard LangChain Message objects."""
    if not history:
        return []
    messages = []
    for msg in history:
        # 1. Native LangChain message object
        if isinstance(msg, (HumanMessage, AIMessage, SystemMessage)):
            messages.append(msg)
            continue

        # 2. Extract role & content from Dict or Pydantic Object
        if isinstance(msg, dict):
            role = msg.get("role")
            content = msg.get("content", "")
            print("dict detected")
        else:
            role = getattr(msg, "role", None)
            content = getattr(msg, "content", "")

        # 3. Map to LangChain classes
        if role in ("user", "human"):
            messages.append(HumanMessage(content=content))
        elif role in ("assistant", "ai"):
            messages.append(AIMessage(content=content))
        elif role == "system":
            messages.append(SystemMessage(content=content))

    return messages


def build_rag_chain(retriever, llm):
    trimmer = trim_messages(
        max_tokens=8,
        strategy="last",
        token_counter=len,
        include_system=True,
        allow_partial=False,
        start_on="human",
    )

    rag_chain = (
        RunnableParallel(
            {
                "context": itemgetter("question") | RunnableLambda(retriever.retrieve),
                "question": itemgetter("question"),
                "chat_history": itemgetter("chat_history"),
            }
        )
        .assign(
            answer=(
                RunnablePassthrough.assign(
                    context=lambda x: format_chunks(x["context"]),
                    chat_history=itemgetter("chat_history")
                    | RunnableLambda(convert_frontend_messages)
                    | trimmer,
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
        llm=get_llm(),
    )