from typing import List
from langchain_core.tracers.stdout import ConsoleCallbackHandler as _ConsoleCallbackHandler

class ConsoleCallbackHandler(_ConsoleCallbackHandler):
    name: str = "lcstackconsole_callback_handler"
    def __init__(self, callbacks: List[str] = None, **kwargs) -> None:
        """Tracer that prints to the console."""
        super().__init__(**kwargs)

        self.callbacks = callbacks or ["llm", "chain", "agent", "retriever", "custom_event", "retry"]

    @property
    def ignore_llm(self) -> bool:
        """Whether to ignore LLM callbacks."""
        return "llm" not in self.callbacks

    @property
    def ignore_retry(self) -> bool:
        """Whether to ignore retry callbacks."""
        return "retry" not in self.callbacks

    @property
    def ignore_chain(self) -> bool:
        """Whether to ignore chain callbacks."""
        return "chain" not in self.callbacks

    @property
    def ignore_agent(self) -> bool:
        """Whether to ignore agent callbacks."""
        return "agent" not in self.callbacks

    @property
    def ignore_retriever(self) -> bool:
        """Whether to ignore retriever callbacks."""
        return "retriever" not in self.callbacks

    @property
    def ignore_chat_model(self) -> bool:
        """Whether to ignore chat model callbacks."""
        return "chat_model" not in self.callbacks and "llm" not in self.callbacks

    @property
    def ignore_custom_event(self) -> bool:
        """Ignore custom event."""
        return "custom_event" not in self.callbacks