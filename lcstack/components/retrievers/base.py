from langchain_core.vectorstores import VectorStore

"""
Args:
    search_type (Optional[str]): Defines the type of search that
        the Retriever should perform.
        Can be "similarity" (default), "mmr", or
        "similarity_score_threshold".
    search_kwargs (Optional[Dict]): Keyword arguments to pass to the
        search function. Can include things like:
            k: Amount of documents to return (Default: 4)
            score_threshold: Minimum relevance threshold
                for similarity_score_threshold
            fetch_k: Amount of documents to pass to MMR algorithm (Default: 20)
            lambda_mult: Diversity of results returned by MMR;
                1 for minimum diversity and 0 for maximum. (Default: 0.5)
            filter: Filter by document metadata
"""
def load_vectorstore_retriever(vectorstore: VectorStore, **kwargs):
    search_type = kwargs.get("search_type", "similarity")
    search_kwargs = kwargs.get("search_kwargs", {
        'k': 4,
        'lambda_mult': 0.5,
        "fetch_k": 20,
    })
    return vectorstore.as_retriever(search_type=search_type, search_kwargs=search_kwargs)