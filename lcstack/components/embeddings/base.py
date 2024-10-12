from typing import Dict, Tuple
from langchain_core.embeddings import Embeddings

SUPPORTED_EMBEDDINGS = Dict[Tuple[str, str], Embeddings]

def create_embeddings(provider: str, tag: str = "embeddings", **kwargs):
    if provider == "openai":
        from langchain_openai.embeddings import OpenAIEmbeddings

        return OpenAIEmbeddings(**kwargs)
    elif provider == "huggingface" or provider == "sentence_transformer":
        from langchain_huggingface import HuggingFaceEmbeddings

        return HuggingFaceEmbeddings(**kwargs)
    elif provider == "ollama":
        from langchain_ollama.embeddings import OllamaEmbeddings

        return OllamaEmbeddings(**kwargs)
    elif SUPPORTED_EMBEDDINGS.get((provider, tag)) is not None:
        embeddings_class = SUPPORTED_EMBEDDINGS.get((provider, tag))
        if not issubclass(embeddings_class, Embeddings):
            raise ValueError(
                f"Embedding provider `{provider}` with tag `{tag}` does not return an instance of Embeddings"
            )
        return embeddings_class(**kwargs)
    else:
        raise NotImplementedError(
            f"Embedding Provider {provider} not supported yet."
            " You can add it to the lcstack.components.embeddings.base.SUPPORTED_EMBEDDINGS manually")
