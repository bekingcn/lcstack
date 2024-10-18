from typing import Any, Dict, Optional, Union

from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import Runnable, RunnablePassthrough
from langchain_core.prompts import BasePromptTemplate, ChatPromptTemplate, MessagesPlaceholder
from langchain_core.language_models import BaseLanguageModel
from langchain_core.retrievers import BaseRetriever, RetrieverOutput, RetrieverOutputLike

from langchain.chains.question_answering import load_qa_chain as _load_qa_chain
from langchain.text_splitter import TextSplitter
from langchain.chains.retrieval import create_retrieval_chain as _create_retrieval_chain
from langchain_core.vectorstores import VectorStoreRetriever
from langchain.chains.combine_documents import create_stuff_documents_chain

from ..retrievers import create_history_aware_retriever
from ..retrievers.base import CONTEXTUALIZE_QUESTION_PROMPT
from ..document_loaders import create_document_loader_chain
from ..utils import keyed_value_runnable, dekey_value_runnable, filter_out_keys_runnable

# Stuff answer question
_stuff_qa_system_prompt = (
    "You are an assistant for question-answering tasks. Use "
    "the following pieces of retrieved context to answer the "
    "question. If you don't know the answer, just say that you "
    "don't know. Use three sentences maximum and keep the answer "
    "concise."
    "\n\n"
    "{context}"
)
STUFF_QA_PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", _stuff_qa_system_prompt),
        MessagesPlaceholder("chat_history", optional=True),
        ("human", "{question}"),
    ]
)
DEFAULT_QUESTION_KEY = "question"

def load_qa_chain(
    llm: BaseLanguageModel,
    chain_type: str = "stuff",
    **kwargs,
) -> Runnable[Dict[str, Any], Any]:
    if chain_type == "stuff":
        qa_prompt = kwargs.get("qa_prompt", None) or STUFF_QA_PROMPT
        question_answer_chain = create_stuff_documents_chain(llm, prompt=qa_prompt, output_parser=StrOutputParser()) | keyed_value_runnable(key="output")
    else:
        # align input_key and output_key
        _chain_kwargs = {**kwargs, "input_key": "context", "output_key": "output"}
        question_answer_chain = _load_qa_chain(
            llm, chain_type=chain_type, **_chain_kwargs
        )

    # invoke input: {"question": ...}, output: {"output": ...}
    return question_answer_chain

def create_retrieval_chain(
    retriever: Union[BaseRetriever, Runnable[dict, RetrieverOutput]],
    combine_docs_chain: Runnable[Dict[str, Any], Union[str | Dict[str, Any]]],
    combine_docs_output_key: Optional[str] = "output",
):
    # invoke input: depends on prompt, default: {"question": ...}, output: {"answer": ...}
    if combine_docs_output_key:
        combine_docs_chain = combine_docs_chain | dekey_value_runnable(key=combine_docs_output_key)
    return _create_retrieval_chain(retriever=retriever, combine_docs_chain=combine_docs_chain)
    # | ( lambda x: {"output": x["answer"]} )
        

def load_conversational_retrieval_chain(
    llm,
    retriever,
    chain_type: str = "stuff",
    qa_prompt: BasePromptTemplate | None = None,
    condense_question_prompt: BasePromptTemplate | None = None,
    condense_question_llm=None,
):
    """
    Create a Conversational Retrieval Chain.
    
    NOTE: we support `staff` chain type as default.
    also support `map_reduce`, `map_rerank` and `refine` with deprecated chain types,
    reference to `MapReduceDocumentsChain`, `MapRerankDocumentsChain` and `RefineDocumentsChain` for more details
    from `https://python.langchain.com/docs/versions/migrating_chains/`

    Args:
        llm: LLM to use
        retriever: Retriever to use
        chain_type: Type of chain to use
        condense_question_prompt: Prompt to use for history-aware retrieval
        condense_question_llm: LLM to use for history-aware retrieval

    Example:
        .. code-block:: python
        inputs = {"input": "..."}
        conversational_retrieval_chain.invoke(inputs, ...)

    """
    qa_chain_kwargs = {"qa_prompt": qa_prompt} if chain_type == "stuff" else {}
    question_answer_chain = load_qa_chain(llm, chain_type=chain_type, **qa_chain_kwargs)
    history_aware_retriever = create_history_aware_retriever(
        llm=condense_question_llm or llm,
        retriever=retriever,
        prompt=condense_question_prompt or CONTEXTUALIZE_QUESTION_PROMPT,
    )
    # Below we use create_stuff_documents_chain to feed all retrieved context
    # into the LLM. or other instances of BaseCombineDocumentsChain.
    # use default output_key="output"
    rag_chain = create_retrieval_chain(
        history_aware_retriever, question_answer_chain
    )
    # invoke input: {"question": ...}, output: {"answer": ...}
    return rag_chain

def _recreate_retriever(retriever: VectorStoreRetriever, search_kwargs, search_type=None):
    if not isinstance(retriever, VectorStoreRetriever):
        raise ValueError("retriever must be a VectorStoreRetriever to modify search_kwargs")
    vs = retriever.vectorstore
    return vs.as_retriever(search_type=search_type or retriever.search_type, search_kwargs={**retriever.search_kwargs, **search_kwargs})

def _retrieve_with_search_kwargs(input_key, retriever) -> RetrieverOutputLike:
    _retriever = retriever
    def _func(input: dict) -> RetrieverOutput:
        search_kwargs = input.get("search_kwargs", None)
        if search_kwargs:
            # change retriever's search_kwargs if input has search_kwargs
            # NOTE: this is a hack to support `filter` at calling time
            # but it's not working with `history_aware_retriever`
            # TODO: any better way to do this?
            retriever = _recreate_retriever(_retriever, search_kwargs)
        else:
            retriever = _retriever
        return retriever.invoke(input[input_key])

    return _func

def load_retrieval_chain(
    llm,
    retriever,
    chain_type: str = "stuff",
    chain_type_kwargs: Dict[str, Any] = {},
    input_key: str = DEFAULT_QUESTION_KEY,
    search_kwargs: Dict[str, Any]={},
):
    # make a copy and append search_kwargs when needed
    # TODO: it's better to pass vectorstore and search_kwargs explicitly, instead of relying on the retriever
    if search_kwargs:
        if isinstance(retriever, VectorStoreRetriever):
            retriever = _recreate_retriever(retriever, search_kwargs)
        else:
            raise ValueError("retriever must be a VectorStoreRetriever to modify search_kwargs")
    retrieval_docs = RunnablePassthrough.assign() | _retrieve_with_search_kwargs(
        input_key=input_key,
        retriever=retriever,
    )
    combine_docs_chain = load_qa_chain(llm, chain_type=chain_type, **chain_type_kwargs)

    # invoke input: {"question": ...}, output: {"answer": ...}
    return create_retrieval_chain(retriever=retrieval_docs, combine_docs_chain=combine_docs_chain)

def load_document_qa_chain(
    llm,
    text_splitter: Optional[TextSplitter] = None,
    chain_type: str = "stuff",
    chain_type_kwargs: Dict[str, Any] = {},
    input_key: str = DEFAULT_QUESTION_KEY,
):
    # TODO: any better way to exclude input_key from invoke input for loader_class?
    load_chain = filter_out_keys_runnable(keys=[input_key]) | create_document_loader_chain(text_splitter=text_splitter)
    combine_docs_chain = load_qa_chain(llm, chain_type=chain_type, **chain_type_kwargs)

    # invoke input: {"question": ...}, output: {"answer": ...}
    return create_retrieval_chain(retriever=load_chain, combine_docs_chain=combine_docs_chain)
