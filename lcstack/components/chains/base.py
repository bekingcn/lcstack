from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import Runnable, RunnablePassthrough
from langchain_core.prompts import BasePromptTemplate, ChatPromptTemplate
from langchain_core.language_models import BaseLanguageModel

from langchain.chains.question_answering import load_qa_chain
from langchain.chains.conversational_retrieval.base import ConversationalRetrievalChain
from langchain.chains.qa_with_sources.retrieval import RetrievalQAWithSourcesChain
from langchain.chains.llm import LLMChain
from langchain.chains.question_answering import load_qa_chain
from langchain.chains.sql_database.query import create_sql_query_chain as _create_sql_query_chain

from langchain_community.chains.llm_requests import LLMRequestsChain
from langchain_community.utilities.sql_database import SQLDatabase

from langchain.text_splitter import TextSplitter

from ..document_loaders import create_document_loader_chain

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
from typing import Any, Dict, List, Optional
from langchain.chains.conversational_retrieval.prompts import CONDENSE_QUESTION_PROMPT
def load_conversational_retrieval_chain(
    llm,
    retriever,
    condense_question_prompt = CONDENSE_QUESTION_PROMPT,
    chain_type: str = "stuff",
    verbose: bool = False,
    condense_question_llm = None,
    combine_docs_chain_kwargs: Optional[Dict] = None,
    **kwargs: Any,
):
    return ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        condense_question_prompt=condense_question_prompt,
        chain_type=chain_type,
        verbose=verbose,
        condense_question_llm=condense_question_llm,
        combine_docs_chain_kwargs=combine_docs_chain_kwargs,
        **kwargs
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
    **kwargs
):
    return RetrievalQAWithSourcesChain.from_chain_type(
        llm=llm,
        chain_type=chain_type,
        retriever=retriever,
        reduce_k_below_max_tokens=reduce_k_below_max_tokens,
        max_tokens_limit=max_tokens_limit,
        chain_type_kwargs=chain_type_kwargs,
        **kwargs
    )

from langchain_core.vectorstores import VectorStore
def _retrieve_with_search_kwargs(input_key, retriever, original_search_kwargs={}):
    original_search_kwargs = original_search_kwargs or {}
    def _func(input):
        search_kwargs = input.get("search_kwargs", None) or {}
        retriever.search_kwargs = {**original_search_kwargs, **search_kwargs}
        return retriever.invoke(input[input_key])
    return _func

def load_retrieval_chain(
    llm,
    retriever,
    chain_type: str = "stuff",
    chain_type_kwargs: Optional[dict] = None,
    search_kwargs=None,
):
    # make a copy and append search_kwargs when needed
    # TODO: it's better to pass vectorstore and search_kwargs explicitly, instead of relying on the retriever
    retriever = retriever.copy()
    original_search_kwargs = retriever.search_kwargs
    search_kwargs = search_kwargs or {}
    original_search_kwargs.update(search_kwargs)
    retrieval_docs = _retrieve_with_search_kwargs(input_key="question", retriever=retriever, original_search_kwargs=original_search_kwargs)
    _chain_kwargs = chain_type_kwargs or {}
    combine_docs_chain = load_qa_chain(llm, chain_type=chain_type, **_chain_kwargs)

    retrieval_chain = (
        RunnablePassthrough.assign(
            input_documents=retrieval_docs, # .with_config(run_name="retrieve_documents")
        ).assign(answer=combine_docs_chain | (lambda x: x["output_text"]))
    ).with_config(run_name="retrieval_chain")

    return retrieval_chain

def load_document_qa_chain(
    llm,
    text_splitter: Optional[TextSplitter]=None,
    chain_type: str = "stuff",
    chain_type_kwargs: Optional[dict] = None,
):
    load_chain = create_document_loader_chain(text_splitter=text_splitter)
    _chain_kwargs = chain_type_kwargs or {}
    combine_docs_chain = load_qa_chain(llm, chain_type=chain_type, **_chain_kwargs)

    doc_qa_chain = (
        RunnablePassthrough.assign(
            input_documents=load_chain.with_config(run_name="load_documents"),
        ).assign(answer=combine_docs_chain | (lambda x: x["output_text"]))
    ).with_config(run_name="document_qa_chain")
    return doc_qa_chain

def load_llm_requests_chain(llm, prompt):
    llm_chain = LLMChain(llm=llm, prompt=prompt)
    chain = LLMRequestsChain(llm_chain=llm_chain)
    return chain

def create_llm_chain(llm: Runnable, prompt: Runnable, return_str: bool = True):
    if return_str:
        return (
            prompt
            | llm
            | StrOutputParser()
        )
    else:
        return (
            prompt
            | llm
        )

def create_sql_query_chain(
    llm: BaseLanguageModel,
    db_uri: str,
    db_engine_args: Optional[dict] = None,
    db_kwargs: Optional[dict] = None,
    prompt: Optional[BasePromptTemplate] = None,
    k: int = 5,
) -> Runnable[Dict[str, Any], str]:
    """Create a chain that generates SQL queries."""
    _engine_args = db_engine_args or {}
    _db_kwargs = db_kwargs or {}
    db = SQLDatabase.from_uri(db_uri, engine_args=_engine_args, **_db_kwargs)
    return _create_sql_query_chain(llm=llm, db=db, prompt=prompt, k=k)

def create_router_chain(
    llm: BaseLanguageModel,
    method: str = "function_calling",
    system_prompt: Optional[str] = None,
    default_members: List[str] = [],
) -> Runnable[Dict[str, Any], Dict[str, Any]]:
    route_system = """Answer the following question.
Route the user's query to either the {members_str} worker. 
Make sure to return ONLY a JSON blob with keys 'destination'.\n\n"""

    route_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", route_system if system_prompt is None else system_prompt),
            ("human", "{query}"),
        ]
    )

    members_str = lambda x: ", ".join(x["members"][:-1]) + " or " + x["members"][-1]
    def router_query_schema(x):
        return {
            "name": "RouteQuery",
            "description": "Route query to destination.",
            "parameters": {
                "type": "object",
                "properties": {
                    "destination": {
                        "enum": x["members"],
                        "type": "string"
                    }
                },
                "required": [
                    "destination"
                ]
            }
        }

    route_chain = (
        RunnablePassthrough.assign(members=lambda x: x.get("members", default_members)).assign(members_str=members_str)
        | (
            lambda x: route_prompt | llm.with_structured_output(schema=router_query_schema(x), method=method, include_raw=False)
        )
    )
    return route_chain