from langchain_community.tools.tavily_search.tool import TavilySearchResults # max_results
from langchain_community.tools import DuckDuckGoSearchResults
from langchain.chains.llm import LLMChain

from lcstack.core.parsers.base import MappingParserArgs, OutputParserType, default_output_parser_args_with_type

from .components.prompts import load_prompt
from .components.tools import create_tool_scrape_webpages
from .components.retrievers import load_vectorstore_retriever
from .components.document_loaders import create_document_loader_chain, create_typed_document_loader_chain
from .components.chat_history import create_chat_history
from .components.indexing import create_indexing_chain
from .components.languge_models import create_llm
from .components.embeddings import create_embeddings
from .components.vectorstores import create_vectorstore
from .components.text_splitters import create_default_text_splitter
from .components.chains import (
    load_document_qa_chain,
    load_llm_requests_chain,
    load_retrieval_chain,
    load_retrieval_qa_with_sources_chain,
    create_llm_chain,
    create_document_loader_chain,
    create_router_chain,
    create_sql_query_chain,
)
from .components.agents import (
    create_hierarchical_team, 
    create_supervisor, 
    react_agent, 
    create_plan_and_execute_agent, 
    create_react_with_plan_agent, 
    create_lc_plan_and_execute_agent,
    create_tools_chat_model,
    create_tool_node
)
from .components.converters import create_data_parser

from .core.parsers import DataType
from .core.component import Component, ComponentType, LcComponent
from .core.models import ComponentParameter

INITIALIZER_NAME_LLM = "llm"
INITIALIZER_NAME_PROMPT = "prompt"
INITIALIZER_NAME_TEXT_SPLITTER = "text_splitter"
INITIALIZER_NAME_TYPED_TEXT_SPLITTER = "typed_text_splitter"
INITIALIZER_NAME_DOCUMENT_LOADER = "document_loader"
INITIALIZER_NAME_TYPED_DOCUMENT_LOADER = "typed_document_loader"
INITIALIZER_NAME_EMBEDDINGS = "embeddings"
INITIALIZER_NAME_VECTORSTORE = "vectorstore"
INITIALIZER_NAME_RETRIEVER = "retriever"
INITIALIZER_NAME_MEMORY = "memory"

INITIALIZER_NAME_LLM_CHAIN = "llm_chain"
INITIALIZER_NAME_RETRIEVAL_CHAIN = "retrieval_chain"
INITIALIZER_NAME_QA_CHAIN = "qa_chain"
INITIALIZER_NAME_CONVERSATIONAL_RETRIEVAL_CHAIN = "conversational_retrieval_chain"
INITIALIZER_NAME_RETRIEVAL_QA_WITH_SOURCES_CHAIN = "retrieval_qa_with_sources_chain"
INITIALIZER_NAME_SQL_QUERY_CHAIN = "sql_query_chain"
INITIALIZER_NAME_LLM_REQUESTS_CHAIN = "llm_requests_chain"
INITIALIZER_NAME_INDEXING_PIPELINE = "indexing_pipeline"

# TODO: seperate components registering from implementing
_COMPONENTS_REGISTRY: dict[str, Component] = {}

def register_component(name, component: Component):
    if name in _COMPONENTS_REGISTRY:
        raise ValueError(f"Component {name} already registered")
    _COMPONENTS_REGISTRY[name] = component
    return component

def get_component(name) -> Component:
    component = _COMPONENTS_REGISTRY.get(name, None)
    if not component:
        raise ValueError(f"Component {name} not found")
    return component

## ===== Register components =======

## helper functions
register_component("data_parser", LcComponent(
    name="data_parser",
    description="Data Parser",
    func_or_class=create_data_parser,
    params={},
    component_type=ComponentType.DataConverter,
    default_output_parser_args=default_output_parser_args_with_type(DataType.pass_through),
))

## common runnables

register_component("llm", LcComponent(
    name="llm",
    description="LLM/Chat Model",
    func_or_class=create_llm,
    params={},
    # this is default value, not neccesary
    default_output_parser_args=default_output_parser_args_with_type(DataType.pass_through),
    component_type=ComponentType.LLM,
))

register_component("prompt", LcComponent(
    name="prompt",
    description="Prompt",
    func_or_class=load_prompt,
    params={},
    component_type=ComponentType.PromptTemplate,
))

# inputs: according to the prompt template
# outputs: str
register_component("llm_chain", LcComponent(
    name="llm_chain",
    description="LLM Chain",
    func_or_class=LLMChain,
    params={},
    component_type=ComponentType.Chain,
    default_output_parser_args=MappingParserArgs(
        output_type=DataType.struct,
        message_key="output"
    )
))

# inputs: according to the prompt template
# outputs: str
register_component("llm_chain_lcel", LcComponent(
    name="llm_chain_lcel",
    description="LLM Chain LCEL",
    func_or_class=create_llm_chain,
    params={},  # str
    component_type=ComponentType.Chain,
    # change the output from str to dict
    default_output_parser_args=MappingParserArgs(
        output_type=DataType.struct,
        message_key="output"
    )
))

# inputs: url
# outputs: {"output": "..."}
register_component("llm_requests_chain", LcComponent(
    name="llm_requests_chain",
    description="LLM Requests Chain",
    func_or_class=load_llm_requests_chain,
    params={},
    inputs=["url"],
    component_type=ComponentType.Chain,
))

# inputs: question
# outputs: dict, answer -> output
register_component("retrieval_chain", LcComponent(
    name="retrieval_chain",
    description="Retrieval Chain",
    func_or_class=load_retrieval_chain,
    params={},
    inputs=["question", "search_kwargs"],
    # re-map the output from dict to dict
    default_output_parser_args=MappingParserArgs(
        output_type=DataType.struct,
        struct_mapping={
            "output": "answer"
        }
    ),
    component_type=ComponentType.Chain,
))

# inputs: question
# outputs: dict, answer -> output, sources
register_component(INITIALIZER_NAME_RETRIEVAL_QA_WITH_SOURCES_CHAIN, LcComponent(
    name=INITIALIZER_NAME_RETRIEVAL_QA_WITH_SOURCES_CHAIN,
    description="Retrieval QA With Sources Chain",
    func_or_class=load_retrieval_qa_with_sources_chain,
    params={},
    inputs=["question"],
    component_type=ComponentType.Chain,
    # re-map the output from dict to dict
    default_output_parser_args=MappingParserArgs(
        output_type=DataType.struct,
        # desired output: `output`, from `answer`
        struct_mapping = {
            "output": "answer",
            "sources": "sources",
        },
    ),
))

# inputs: file_path, question
# outputs: dict, answer -> output
register_component("document_qa_chain", LcComponent(
    name="document_qa_chain",
    description="Document QA Chain",
    func_or_class=load_document_qa_chain,
    params={},
    # here use `query` as inputs instead of `question`
    inputs={
        "question": "query", 
        "file_path": "file_path"
    },
    component_type=ComponentType.Chain,
    default_output_parser_args=MappingParserArgs(
        output_type=DataType.struct,
        # desired output: `output`, from `answer`
        struct_mapping = {
            "output": "answer",
        },
    ),
))

# inputs: question, table_names_to_use (optional, list[str])
# outputs: str -> dict
register_component("sql_query_chain", LcComponent(
    name="sql_query_chain",
    description="SQL Query Chain",
    func_or_class=create_sql_query_chain,
    params={},
    inputs=["question", "table_names_to_use"],
    component_type=ComponentType.Chain,
    default_output_parser_args=MappingParserArgs(
        output_type=DataType.struct,
        # desired output: `output`, from plain text
        message_key = "output",
    ),
))

# inputs: dict, query, optional members (list[str])
# outputs: dict, destination, not mapped
register_component("router_chain", LcComponent(
    name="router_chain",
    description="Router Chain",
    func_or_class=create_router_chain,
    params={},
    inputs=["query", "members"],
    component_type=ComponentType.Chain,
))

# Agents

## Helper Components
""" Notes from ToolNode docstring
Important:
    - The state MUST contain a list of messages.
    - The last message MUST be an `AIMessage`.
    - The `AIMessage` MUST have `tool_calls` populated.

based on this, tool_node's upstream must be a `ChatModel` which binds same tools
"""
register_component("tool_node", LcComponent(
    name="tool_node",
    description="Tool Node",
    func_or_class=create_tool_node,
    params={},
    component_type=ComponentType.ToolNode,
))

# inputs: str, or dict {"messages": ...}
# outputs: dict, answer -> output
register_component("react_agent", LcComponent(
    name="react_agent",
    description="React Agent",
    func_or_class=react_agent,
    params={
        "messages_modifier": ComponentParameter(type="str"), # str, SystemMessage, Runnable, Callable(not supported)
        "tools": ComponentParameter(type="list[Tool]", component_type="Tool"),
    },
    inputs=["messages"],
    component_type=ComponentType.Chain,
    default_output_parser_args=MappingParserArgs(
        output_type=DataType.struct,
        struct_mapping = {
            "output": "answer",
        },
    ),
))

# inputs: dict, `messages`
# outputs: dict, next, not mapped
register_component("supervisor_agent", LcComponent(
    name="supervisor_agent",
    description="Supervisor Agent",
    func_or_class=create_supervisor,
    params={},
    inputs=["messages"],
    component_type=ComponentType.Chain,
))

# TODO: reset the inputs, for now plain text
# inputs: str
# outputs: dict, next, not mapped
register_component("hierarchical_team", LcComponent(
    name="hierarchical_team",
    description="Hierarchical Team",
    func_or_class=create_hierarchical_team,
    params={},
    inputs= "input",    # str to {"input": "..."}
    component_type=ComponentType.Chain,
    # re-map the plain text to dict with key `output`
    default_output_parser_args=MappingParserArgs(
        output_type=DataType.struct,
        message_key = "output",
    ),
))

# with planning and replanning, executing with agents
# inputs: input
# outputs: dict, response -> output
register_component("plan_and_execute", LcComponent(
    name="plan_and_execute",
    description="Plan and Execute",
    func_or_class=create_plan_and_execute_agent,
    params={},
    inputs=["input"],
    component_type=ComponentType.Chain,
    default_output_parser_args=MappingParserArgs(
        output_type=DataType.struct,
        struct_mapping = {
            "output": "response",
        },
    ),
))

# with replanning, using react_agent
# inputs: input
# outputs: dict, response -> output
register_component("react_plan_and_execute", LcComponent(
    name="react_plan_and_execute",
    description="React Plan and Execute",
    func_or_class=create_react_with_plan_agent,
    params={},
    inputs=["input"],
    component_type=ComponentType.Chain,
    default_output_parser_args=MappingParserArgs(
       output_type=DataType.struct,
        struct_mapping = {
            "output": "response",
        },
    ),
))

# without replanning, without structured calling
# inputs: input, or specify by `input_key`
# outputs: dict, response (or specify by `output_key`) -> output
register_component("lc_plan_and_execute", LcComponent(
    name="lc_plan_and_execute",
    description="LC Plan and Execute",
    func_or_class=create_lc_plan_and_execute_agent,
    params={},
    inputs=["input"],
    component_type=ComponentType.Chain,
    default_output_parser_args=MappingParserArgs(
        output_type=DataType.struct,
        struct_mapping = {
            "output": "response",
        },
    ),
))

# TOOLS, all tools' outputs will be passed through
register_component("tavily_search", LcComponent(
    name="tavily_search",
    description="Tavily Search Results",
    func_or_class=TavilySearchResults,
    params={},
    component_type=ComponentType.Tool,
))

register_component("ddg_search", LcComponent(
    name="ddg_search",
    description="DuckDuckGo Search Results",
    func_or_class=DuckDuckGoSearchResults,
    params={},
    component_type=ComponentType.Tool,
))

register_component("scrape_webpages", LcComponent(
    name="scrape_webpages",
    description="Scrape Webpages",
    func_or_class=create_tool_scrape_webpages,
    params={},
    component_type=ComponentType.Tool,
))

# Data Pipeline
# inputs: file_path, extensions (optional)
# outputs: dict with `documents`, not mapped
register_component(INITIALIZER_NAME_DOCUMENT_LOADER, LcComponent(
    name=INITIALIZER_NAME_DOCUMENT_LOADER,
    description="Document Loader Chain",
    func_or_class=create_document_loader_chain,
    params={},
    # not check inputs and outputs for now
    inputs=["file_path", "extensions"],
    component_type=ComponentType.DocumentLoader,
))

# inputs: according to the document loader type
# same as document loader
register_component(INITIALIZER_NAME_TYPED_DOCUMENT_LOADER, LcComponent(
    name=INITIALIZER_NAME_TYPED_DOCUMENT_LOADER,
    description="Typed Document Loader Chain",
    func_or_class=create_typed_document_loader_chain,
    params={},
    # not check inputs and outputs for now
    inputs=["file_path", "extensions"],  # same as document loader class
    component_type=ComponentType.DocumentLoader,
))

register_component(INITIALIZER_NAME_TEXT_SPLITTER, LcComponent(
    name=INITIALIZER_NAME_TEXT_SPLITTER,
    description="Text Splitter",
    func_or_class=create_default_text_splitter,
    params={},
    component_type=ComponentType.TextSplitter,
))

register_component(INITIALIZER_NAME_EMBEDDINGS, LcComponent(
    name=INITIALIZER_NAME_EMBEDDINGS,
    description="Embeddings",
    func_or_class=create_embeddings,
    params={},
    component_type=ComponentType.Embeddings,
))

register_component(INITIALIZER_NAME_VECTORSTORE, LcComponent(
    name=INITIALIZER_NAME_VECTORSTORE,
    description="Vector Store",
    func_or_class=create_vectorstore,
    params={},
    component_type=ComponentType.VectorStore,
))

register_component(INITIALIZER_NAME_RETRIEVER, LcComponent(
    name=INITIALIZER_NAME_RETRIEVER,
    description="Retriever",
    func_or_class=load_vectorstore_retriever,
    params={},
    inputs=[],
    component_type=ComponentType.Retriever,
))

register_component(INITIALIZER_NAME_INDEXING_PIPELINE, LcComponent(
    name=INITIALIZER_NAME_INDEXING_PIPELINE,
    description="Indexing Pipeline",
    func_or_class=create_indexing_chain,
    params={},
    inputs=["file_path", "extensions"],
    component_type=ComponentType.Chain,
))

# Memory

register_component("chat_history", LcComponent(
    name="chat_history",
    description="Chat History",
    func_or_class=create_chat_history,
    params={},
    component_type=ComponentType.ChatMessageHistory,
))

# Workflow

# Graph
from .graph import create_workflow_agent, WorkflowType
import functools
register_component("conditional_graph", LcComponent(
    name="conditional_graph",
    description="Conditional Graph",
    func_or_class=create_workflow_agent,
    params={},
    component_type=ComponentType.StateGraph,
))