from typing import Any, Dict, List, Optional, Union

from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import Runnable, RunnablePassthrough
from langchain_core.prompts import BasePromptTemplate, ChatPromptTemplate, MessagesPlaceholder
from langchain_core.language_models import BaseLanguageModel
from langchain_core.retrievers import BaseRetriever, RetrieverOutput, RetrieverOutputLike

from langchain.chains.question_answering import load_qa_chain as _load_qa_chain
from langchain.chains.conversational_retrieval.base import ConversationalRetrievalChain
from langchain.chains.qa_with_sources.retrieval import RetrievalQAWithSourcesChain
from langchain.chains.llm import LLMChain
from langchain.chains.sql_database.query import (
    create_sql_query_chain as _create_sql_query_chain,
)
from langchain_community.chains.llm_requests import LLMRequestsChain
from langchain_community.utilities.sql_database import SQLDatabase
from langchain.text_splitter import TextSplitter
from langchain.chains.conversational_retrieval.prompts import CONDENSE_QUESTION_PROMPT
from langchain.chains.retrieval import create_retrieval_chain as _create_retrieval_chain
from langchain_core.vectorstores import VectorStoreRetriever
from langchain.chains.combine_documents import create_stuff_documents_chain

from ..retrievers import create_history_aware_retriever
from ..retrievers.base import CONTEXTUALIZE_QUESTION_PROMPT
from ..document_loaders import create_document_loader_chain

# TODO: deprecated, remove it
"""
    ConversationalRetrievalChain.from_llm:
        llm: BaseLanguageModel,
        retriever: BaseRetriever,
        condense_question_prompt: BasePromptTemplate = CONDENSE_QUESTION_PROMPT,
        chain_type: str = "stuff",
        verbose: bool = False,
        condense_question_llm: Optional[BaseLanguageModel] = None,
        combine_docs_chain_kwargs: Optional[Dict] = None,
"""
def load_conversational_retrieval_chain_deprecated(
    llm,
    retriever,
    condense_question_prompt=CONDENSE_QUESTION_PROMPT,
    chain_type: str = "stuff",
    verbose: bool = False,
    condense_question_llm=None,
    combine_docs_chain_kwargs: Optional[Dict] = None,
    **kwargs: Any,
):
    # invoke input: {"input": ...}, output: {"answer": ...}
    return ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        condense_question_prompt=condense_question_prompt,
        chain_type=chain_type,
        verbose=verbose,
        condense_question_llm=condense_question_llm,
        combine_docs_chain_kwargs=combine_docs_chain_kwargs,
        **kwargs,
    )


"""
BaseQAWithSourcesChain.from_chain_type:
    llm: BaseLanguageModel,
    chain_type: str = "stuff",
    chain_type_kwargs: Optional[dict] = None,
    **kwargs: Any,

RetrievalQAWithSourcesChain:
    retriever: BaseRetriever = Field(exclude=True)
    reduce_k_below_max_tokens: bool = False
    max_tokens_limit: int = 3375
"""


def load_retrieval_qa_with_sources_chain(
    llm,
    retriever,
    chain_type: str = "stuff",
    chain_type_kwargs: Optional[dict] = None,
    reduce_k_below_max_tokens: bool = False,
    max_tokens_limit: int = 3375,
    **kwargs,
):
    return RetrievalQAWithSourcesChain.from_chain_type(
        llm=llm,
        chain_type=chain_type,
        retriever=retriever,
        reduce_k_below_max_tokens=reduce_k_below_max_tokens,
        max_tokens_limit=max_tokens_limit,
        chain_type_kwargs=chain_type_kwargs,
        **kwargs,
    )