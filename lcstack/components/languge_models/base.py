from typing import List, Dict, Tuple, Literal

from langchain_core.language_models import BaseLanguageModel, BaseChatModel
from langchain_core.messages import BaseMessage
from langchain_core.tools import BaseTool
from langgraph.prebuilt import ToolNode

SUPPORTED_LANGUAGE_MODELS = Dict[Tuple[str, str], BaseLanguageModel]

def create_llm(
    provider: str,
    tag: str = "chat",
    tools: List[BaseTool] | ToolNode | None = None,
    tool_choice: dict | str | Literal["auto", "any", "none"] | bool | None = None,
    **kwargs,
) -> BaseLanguageModel[str | BaseMessage]:
    if provider == "openai" and tag == "chat":
        from langchain_openai.chat_models.base import ChatOpenAI

        lang_model_class = ChatOpenAI
    elif provider == "openai" and tag == "llm":
        from langchain_openai.llms.base import OpenAI

        lang_model_class = OpenAI
    elif provider == "groq" and tag == "chat":
        from langchain_groq.chat_models import ChatGroq

        lang_model_class = ChatGroq
    elif provider == "ollama" and tag == "llm":
        from langchain_ollama.llms import OllamaLLM

        lang_model_class = OllamaLLM
    elif provider == "ollama" and tag == "chat":
        from langchain_ollama.chat_models import ChatOllama

        lang_model_class = ChatOllama
    elif provider == "genai" and tag == "llm":
        from langchain_google_genai.llms import GoogleGenerativeAI

        lang_model_class = GoogleGenerativeAI
    elif provider == "genai" and tag == "chat":
        from langchain_google_genai.chat_models import ChatGoogleGenerativeAI

        lang_model_class = ChatGoogleGenerativeAI
    elif SUPPORTED_LANGUAGE_MODELS.get((provider, tag)) is not None:
        lang_model_class = SUPPORTED_LANGUAGE_MODELS.get((provider, tag))
        if not issubclass(lang_model_class, BaseLanguageModel):
            raise ValueError(
                f"Language Model Provider `{provider}` with tag `{tag}` does not return an subclass of BaseLanguageModel"
            )
    else:
        raise NotImplementedError(
            f"Language Model Provider `{provider}` with tag `{tag}` not supported yet."
            " You can add it to the lcstack.components.languge_models.base.SUPPORTED_LANGUAGE_MODELS manually.")

    llm = lang_model_class(**kwargs)
    if tools:
        if (
            isinstance(llm, BaseChatModel)
            and llm.bind_tools is not BaseChatModel.bind_tools
        ):
            if isinstance(tools, ToolNode):
                tool_classes = list(tools.tools_by_name.values())
            else:
                tool_classes = tools
            llm = llm.bind_tools(tool_classes, tool_choice=tool_choice)
        else:
            raise NotImplementedError(
                f"Language Model Provider {provider} with tag {tag} does not support bind_tools."
            )

    return llm

from langchain_core.utils import xor_args
from ..utils import keyed_value_runnable, dekey_value_runnable
from ..output_parser import create_output_parser, SUPPORTED_OUTPUT_PARSERS

@xor_args(("llm", "provider"))
def create_llm_node(
        llm: BaseLanguageModel | None = None, 
        provider: str | None = None, 
        tag: str = "chat",
        tools: List[BaseTool] | ToolNode | None = None,
        tool_choice: dict | str | Literal["auto", "any", "none"] | bool | None = None,
        input_key: str = "messages", 
        output_key: str = "output", 
        output_type: str = "str",
        **kwargs
    ):
    if llm is None:
        llm = create_llm(provider, tag=tag, tools=tools, tool_choice=tool_choice, **kwargs)
    if output_type in SUPPORTED_OUTPUT_PARSERS:
        post_parser = create_output_parser(type=output_type, output_key=output_key)
    elif output_type == "message":
        post_parser = keyed_value_runnable(key=output_key)
    elif output_type == "messages":
        # TODO: improve this
        post_parser = (lambda x: [x]) | keyed_value_runnable(key=output_key)
    else:
        raise ValueError(f"Invalid output type: {output_type}")

    
    return (dekey_value_runnable(key=input_key) | llm | post_parser).with_config(name="llm_node")

