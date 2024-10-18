from langchain_core.vectorstores import VectorStore, VectorStoreRetriever
from langchain_core.retrievers import RetrieverLike, RetrieverOutputLike
from langchain_core.language_models import LanguageModelLike
from langchain_core.utils import xor_args
from langchain.prompts import BasePromptTemplate, PromptTemplate, ChatPromptTemplate, MessagesPlaceholder
from langchain.chains.history_aware_retriever import create_history_aware_retriever as _create_history_aware_retriever

from langchain.chains.conversational_retrieval.prompts import CONDENSE_QUESTION_PROMPT

def load_vectorstore_retriever(vectorstore: VectorStore, search_type: str="similarity", search_kwargs: dict = {}) -> VectorStoreRetriever:
    """
    Return VectorStoreRetriever initialized from a VectorStore with the specified search type.

    Args:
        vectorstore (VectorStore): VectorStore to use for retrieval.
        search_type (Optional[str]): Defines the type of search that
            the Retriever should perform.
            Can be "similarity" (default), "mmr", or
            "similarity_score_threshold".
        search_kwargs (Optional[Dict]): Keyword arguments to pass to the
            search function. Can include things like:
                k: Amount of documents to return (Default: 4)
                score_threshold: Minimum relevance threshold
                    for similarity_score_threshold
                fetch_k: Amount of documents to pass to MMR algorithm (Default: 20)
                lambda_mult: Diversity of results returned by MMR;
                    1 for minimum diversity and 0 for maximum. (Default: 0.5)
                filter: Filter by document metadata
    """
    return vectorstore.as_retriever(
        search_type=search_type, search_kwargs=search_kwargs
    )

# TODO: which prompt?
# String prompt
_template = """Given the following conversation and a follow up question, rephrase the follow up question to be a standalone question, in its original language.

Chat History:
{chat_history}
Follow Up Input: {question}
Standalone question:"""
CONDENSE_QUESTION_PROMPT = PromptTemplate.from_template(_template)

# Contextualize question, chat prompt
_contextualize_q_system_prompt = (
    "Given a chat history and the latest user question "
    "which might reference context in the chat history, "
    "formulate a standalone question which can be understood "
    "without the chat history. Do NOT answer the question, just "
    "reformulate it if needed and otherwise return it as is."
)
CONTEXTUALIZE_QUESTION_PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", _contextualize_q_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{question}"),
    ]
)

@xor_args(("retriever", "vectorstore"))
def create_history_aware_retriever(
        llm: LanguageModelLike, 
        retriever: RetrieverLike | None=None, 
        vectorstore: VectorStore | None=None, 
        search_type: str="similarity", 
        search_kwargs: dict = {}, 
        prompt: BasePromptTemplate = CONTEXTUALIZE_QUESTION_PROMPT
) -> RetrieverOutputLike:
    """
    Create a history-aware retriever as langchain's `create_history_aware_retriever`.
    
    Args:
        llm: Language model to use for generating a search term given chat history
        retriever: RetrieverLike object that takes a string as input and outputs
            a list of Documents, optional.
        vectorstore: VectorStore to use for retrieval, if retriever is not specified.
        search_type: Defines the type of search that the Retriever should perform.
            Can be "similarity" (default), "mmr", or "similarity_score_threshold".
        search_kwargs: Keyword arguments to pass to the search function. Can include things like:
            k: Amount of documents to return (Default: 4)
            score_threshold: Minimum relevance threshold
                for similarity_score_threshold
            fetch_k: Amount of documents to pass to MMR algorithm (Default: 20)
            lambda_mult: Diversity of results returned by MMR;
                1 for minimum diversity and 0 for maximum. (Default: 0.5)
            filter: Filter by document metadata
    """

    if not retriever:
        retriever = load_vectorstore_retriever(vectorstore, search_type=search_type, search_kwargs=search_kwargs)
    return _create_history_aware_retriever(llm=llm, retriever=retriever, prompt=prompt or rephrase_prompt)