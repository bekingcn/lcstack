from typing import List, Dict, Tuple, Callable
from langchain_core.vectorstores import VectorStore


SUPPORTED_VECTORSTORES = Dict[Tuple[str, str], Callable[..., VectorStore]]

def create_vectorstore(
    provider: str, tag: str = "vectorstore", delete_existing: bool = False, **kwargs
):
    if provider == "chromadb":
        from langchain_community.vectorstores.chroma import Chroma

        db = Chroma(**kwargs)
        if delete_existing:
            db.delete_collection()
        return db
    elif provider == "duckdb":
        from langchain_community.vectorstores.duckdb import DuckDB

        db = DuckDB(**kwargs)
        if delete_existing:
            db.delete()
        return db
    elif provider == "faiss":
        from langchain_community.vectorstores.faiss import (
            FAISS,
            dependable_faiss_import,
        )
        from langchain.docstore import InMemoryDocstore
        from langchain_community.vectorstores.utils import DistanceStrategy

        if "index" not in kwargs:
            faiss = dependable_faiss_import()
            distance_strategy = kwargs.get(
                "distance_strategy", DistanceStrategy.MAX_INNER_PRODUCT
            )
            # have to embed a test document before initializing index
            embedding = kwargs.get("embedding")
            embeddings: List[List[float]] = []
            if embedding is not None:
                embeddings = embedding.embed_documents(["test"])
            else:
                raise ValueError(
                    "An embedding must be provided to initialize FAISS index."
                )
            if distance_strategy == DistanceStrategy.MAX_INNER_PRODUCT:
                index = faiss.IndexFlatIP(len(embeddings[0]))
            else:
                # Default to L2, currently other metric types not initialized.
                index = faiss.IndexFlatL2(len(embeddings[0]))
            kwargs["index"] = index
        if "docstore" not in kwargs:
            kwargs["docstore"] = InMemoryDocstore()
        db = FAISS(**kwargs)
        if delete_existing:
            # TODO: Implement delete
            pass
        return db
    elif SUPPORTED_VECTORSTORES.get((provider, tag)) is not None:
        vectorstore_callable = SUPPORTED_VECTORSTORES.get((provider, tag))
        if not isinstance(vectorstore_callable, Callable):
            raise ValueError(
                f"Vectorstore provider `{provider}` with tag `{tag}` does not return an Callable, which returns an instance of VectorStore"
            )
        if delete_existing:
            kwargs["delete_existing"] = True
        vectorstore = vectorstore_callable(**kwargs)
        if not isinstance(vectorstore, VectorStore):
            raise ValueError(
                f"Vectorstore provider `{provider}` with tag `{tag}` does not return an instance of VectorStore"
            )
        return vectorstore
    else:
        raise NotImplementedError(
            f"Vectorstore Provider {provider} not supported yet."
            " You can add it to the lcstack.components.vectorstores.base.SUPPORTED_VECTORSTORES manually")
