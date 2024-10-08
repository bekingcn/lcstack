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
    else:
        raise NotImplementedError(f"Embedding Provider {provider} not supported yet.")
