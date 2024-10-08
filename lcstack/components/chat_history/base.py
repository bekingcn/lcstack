from langchain_core.chat_history import BaseChatMessageHistory

GLOABLE_INMEMORY_CHAT_HISTORY_FACTORY = {}


class ChatHistoryFactory:
    def __init__(self, provider: str, **kwargs):
        self.provider = provider
        self.kwargs = kwargs

    def get_by_session_id(self, session_id: str) -> BaseChatMessageHistory:
        return _create_chat_history_internal(self.provider, session_id, **self.kwargs)


def create_chat_history(provider: str, **kwargs):
    """create a chat history factory which delegates to the provider's implementation

    Args:
        provider (str): the provider to use. Currently supports 'cassandra', 'file' and 'memory'
        kwargs (dict): the kwargs to pass to the provider's ChatMessageHistory

    Returns:
        ChatHistoryFactory
    """

    return ChatHistoryFactory(provider, **kwargs)


def _create_chat_history_internal(provider: str, session_id: str, **kwargs):
    if provider == "cassandra":
        from langchain_community.chat_message_histories import (
            CassandraChatMessageHistory,
        )

        try:
            import cassio
        except ImportError:
            raise ImportError(
                "Could not import cassio integration package. "
                "Please install it with `pip install cassio`."
            )

        from uuid import UUID

        database_ref = kwargs.get("database_ref")
        token = kwargs.get("token")
        cluster_kwargs = kwargs.get("cluster_kwargs")
        username = kwargs.get("username")
        table_name = kwargs.get("table_name")
        keyspace = kwargs.get("keyspace")

        try:
            UUID(database_ref)
            is_astra = True
        except ValueError:
            is_astra = False
            if "," in database_ref:
                # use a copy because we can't change the type of the parameter
                database_ref = database_ref.split(",")

        if is_astra:
            cassio.init(
                database_id=database_ref,
                token=token,
                cluster_kwargs=cluster_kwargs,
            )
        else:
            cassio.init(
                contact_points=database_ref,
                username=username,
                password=token,
                cluster_kwargs=cluster_kwargs,
            )

        return CassandraChatMessageHistory(
            session_id=session_id,
            table_name=table_name,
            keyspace=keyspace,
        )
    elif provider == "file":
        import os
        from pathlib import Path
        from langchain_community.chat_message_histories import FileChatMessageHistory

        dir_path = kwargs.get("dir_path")
        if not dir_path:
            raise ValueError("dir_path is required for provider 'file'")
        dir_path = Path(dir_path)
        if not dir_path.exists():
            dir_path.mkdir(parents=True)
        file_path = os.path.join(dir_path, f"{session_id}.json")

        return FileChatMessageHistory(file_path=file_path)
    elif provider == "memory":
        from langchain_core.chat_history import InMemoryChatMessageHistory

        if session_id in GLOABLE_INMEMORY_CHAT_HISTORY_FACTORY:
            return GLOABLE_INMEMORY_CHAT_HISTORY_FACTORY[session_id]
        else:
            GLOABLE_INMEMORY_CHAT_HISTORY_FACTORY[session_id] = (
                InMemoryChatMessageHistory()
            )
            return GLOABLE_INMEMORY_CHAT_HISTORY_FACTORY[session_id]
