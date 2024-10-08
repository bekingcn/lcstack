from typing import List, Literal

from langchain_core.language_models import BaseLanguageModel, BaseChatModel
from langchain_core.messages import BaseMessage
from langchain_core.tools import BaseTool
from langgraph.prebuilt import ToolNode


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
    else:
        raise NotImplementedError("Language Model Provider not supported yet.")

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
