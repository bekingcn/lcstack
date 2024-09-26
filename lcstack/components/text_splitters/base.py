from typing import List
from langchain.text_splitter import RecursiveCharacterTextSplitter, TextSplitter
from langchain_core.documents import Document
from langchain_core.runnables import Runnable

def create_typed_text_splitter(
        text_splitter_class, **kwargs
) -> Runnable[List[Document], List[Document]]: # type: ignore
    _text_splitter_class = text_splitter_class
    if isinstance(text_splitter_class, str):
        # import a type from a string
        import importlib
        text_splitter_class = getattr(importlib.import_module("langchain.text_splitter"), text_splitter_class, None)
        if not text_splitter_class:
            parts = _text_splitter_class.rsplit(".", 1)
            if len(parts) == 2:
                module, cls = parts
                text_splitter_class = getattr(importlib.import_module(module), cls, None)
            else:
                text_splitter_class = globals().get(text_splitter_class, None)
    if not text_splitter_class:
        raise ValueError(f"Unknown text splitter: {_text_splitter_class}")
    if not issubclass(text_splitter_class, TextSplitter):
        raise ValueError(f"Text splitter class {_text_splitter_class} is not a subclass of TextSplitter")
    return text_splitter_class(**kwargs)

def create_default_text_splitter(**kwargs) -> Runnable[List[Document], List[Document]]:
    return create_typed_text_splitter(RecursiveCharacterTextSplitter, **kwargs)