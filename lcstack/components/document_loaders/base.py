from typing import List, Tuple, Type, Dict, Any

from langchain_community.document_loaders import (
    CSVLoader,
    EverNoteLoader,
    PyMuPDFLoader,
    TextLoader,
    UnstructuredEmailLoader,
    UnstructuredHTMLLoader,
    UnstructuredMarkdownLoader,
    UnstructuredODTLoader,
    UnstructuredPowerPointLoader,
    UnstructuredWordDocumentLoader,
)
from langchain_community.document_loaders.base import BaseLoader
from langchain.docstore.document import Document
from langchain_core.runnables import RunnablePassthrough, Runnable
from langchain_text_splitters import TextSplitter

from .epub_loader import EpubLibEpubLoader


LOADER_MAPPING: Dict[str, Tuple[Type[BaseLoader], Dict[str, Any]]] = {
    # NOTE: this is a config for book csv
    ".bookcsv": (
        CSVLoader,
        {
            "metadata_columns": [
                "name",
                "id",
                "type",
                "topic",
                "author",
                "language",
                "sn",
                "cover",
            ]
        },
    ),
    ".csv": (CSVLoader, {}),
    # ".docx": (Docx2txtLoader, {}),
    ".doc": (UnstructuredWordDocumentLoader, {}),
    ".docx": (UnstructuredWordDocumentLoader, {}),
    ".enex": (EverNoteLoader, {}),
    ".eml": (UnstructuredEmailLoader, {}),
    # ".epub": (UnstructuredEPubLoader, {"mode": "single"}),  ## mode="elements", doc will be splited into mutil pages
    ".epub": (EpubLibEpubLoader, {"bodywidth": 0, "load_images": False}),
    ".html": (UnstructuredHTMLLoader, {}),
    ".md": (UnstructuredMarkdownLoader, {}),
    ".odt": (UnstructuredODTLoader, {}),
    ".pdf": (PyMuPDFLoader, {}),
    ".ppt": (UnstructuredPowerPointLoader, {}),
    ".pptx": (UnstructuredPowerPointLoader, {}),
    ".txt": (TextLoader, {"encoding": "utf8"}),
    # Add more mappings for other file extensions and loaders as needed
}


class GenericDocumentLoader(BaseLoader):
    def __init__(self, file_path: str, extensions: list[str] = [], **kwargs: Any):
        self.kwargs = kwargs
        self.file_path = file_path
        self.extensions = extensions

    def load(self) -> List[Document]:
        import os

        # TODO: support multiple files at once
        if os.path.isfile(self.file_path):
            file_paths = [self.file_path]
        else:
            file_paths: List[str] = []
            for root, _, files in os.walk(self.file_path):
                for file in files:
                    file_paths.append(os.path.join(root, file))
        documents: List[Document] = []
        for file_path in file_paths:
            ext = "." + file_path.rsplit(".", 1)[-1]
            if self.extensions and ext not in self.extensions:
                continue
            if ext in LOADER_MAPPING:
                loader_class, loader_args = LOADER_MAPPING[ext]
                loader = loader_class(file_path, **loader_args) # type: ignore
                new_documents = loader.load()
                # for doc in new_documents:
                #     doc.metadata["source"] = file_path
                documents.extend(new_documents)
            else:
                raise ValueError(f"Unsupported file extension '{ext}'")
        return documents

    # NOTE: this works globally
    @classmethod
    def add_loader(cls, extension: str, loader_class: Type[BaseLoader], **kwargs: Any):
        LOADER_MAPPING[extension] = (loader_class, kwargs or {})


def create_typed_document_loader_chain(
    loader_class: Type[BaseLoader] | str,
    text_splitter: TextSplitter | None=None,
) -> Runnable[Dict[str, Any], List[Document]]:  # type: ignore
    _loader_class = None
    if isinstance(loader_class, str):
        # import a type from a string
        import importlib
        _loader_class = getattr(
            importlib.import_module("langchain_community.document_loaders"),
            loader_class,
            None,
        )
        if not _loader_class:
            parts = loader_class.rsplit(".", 1)
            if len(parts) == 2:
                module, cls = parts
                _loader_class = getattr(importlib.import_module(module), cls, None)
            else:
                _loader_class = globals().get(loader_class, None)
    if not _loader_class:
        raise ValueError(f"Unknown document loader: {loader_class}")
    if not issubclass(_loader_class, BaseLoader):
        raise ValueError(
            f"Document loader {_loader_class} is not a subclass of BaseLoader"
        )
    return RunnablePassthrough.assign() | (
        lambda x: text_splitter.split_documents(_loader_class(**x).load())
        if text_splitter
        else _loader_class(**x).load()
    )


def create_document_loader_chain(
    text_splitter: TextSplitter | None=None,
) -> Runnable[Dict[str, Any], List[Document]]:
    return create_typed_document_loader_chain(GenericDocumentLoader, text_splitter)
    # return (
    #     RunnablePassthrough.assign(**kwargs)
    #     | (
    #         lambda x: GenericDocumentLoader(file_path=x["file_path"], extensions=x.get("extensions", None), **kwargs).load()
    #     )
    # )
