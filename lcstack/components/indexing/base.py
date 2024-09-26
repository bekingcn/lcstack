from typing import List
import uuid
from ..document_loaders.base import GenericDocumentLoader
from ..document_loaders import create_document_loader_chain

# deprecated, use create_indexing_chain instead. removed in the future
def create_indexing_pipeline(file_path: str, vectorstore, text_splitter=None, extensions: List[str]=None, **kwargs):
    def _func():
        loader_kwargs = {}
        loader_kwargs["file_path"] = file_path
        loader_kwargs["extensions"] = extensions
        db = vectorstore
        document_loader = GenericDocumentLoader(**loader_kwargs)
        # we did not set a text_splitter here, but it will give a default one
        # RecursiveCharacterTextSplitter, all with default parameters (chunck-size=4000, chunk_overlap=200)
        documents = document_loader.load()
        last_source = None
        # group docs by source
        docs_by_source = {}
        for doc in documents:
            source = doc.metadata["source"]
            if source != last_source:
                docs_by_source[source] = []
                last_source = source
            docs_by_source[source].append(doc)
        for source, docs in docs_by_source.items():
            file_id = str(uuid.uuid4())
            for doc in docs:
                doc.metadata["file_id"] = file_id
            if text_splitter:
                docs = text_splitter.split_documents(docs)
            db.add_documents(docs)
        return documents
    return _func


from langchain_core.runnables import RunnablePassthrough, Runnable
from langchain.docstore.document import Document
from typing import Dict, Any, Union
from typing_extensions import TypedDict

class IndexingInput(TypedDict):
    """Input for a Indexing Chain."""

    file_path: str


class IndexingInputWithExtensions(TypedDict):
    """Input for a Indexing Chain."""

    file_path: str
    extensions: List[str]

# TOOD: register this as a chain initializer
def create_indexing_chain(
        vectorstore, document_loader_chain=None, text_splitter=None
    ) -> Runnable[Union[IndexingInput, IndexingInputWithExtensions, Dict[str, Any]], List[Document]]:
    def _func(documents):
        print(documents)
        last_source = None
        # group docs by source
        docs_by_source = {}
        for doc in documents:
            source = doc.metadata["source"]
            if source != last_source:
                docs_by_source[source] = []
                last_source = source
            docs_by_source[source].append(doc)
        for source, docs in docs_by_source.items():
            file_id = str(uuid.uuid4())
            for doc in docs:
                doc.metadata["file_id"] = file_id
            if text_splitter:
                docs = text_splitter.split_documents(docs)
            vectorstore.add_documents(docs)
        return {"documents": documents}
    inputs = {
        "file_path": lambda x: x["file_path"],
        "extensions": lambda x: x.get("extensions", None)
    }
    document_loader_chain = document_loader_chain or create_document_loader_chain()
    return (
        RunnablePassthrough.assign(**inputs)
        | document_loader_chain
        | _func
    )