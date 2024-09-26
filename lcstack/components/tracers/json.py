
import uuid

from .base import SerializedCallbackHandler

class JsonCallbackHandler(SerializedCallbackHandler):
    """Tracer that prints to the console."""

    name: str = "json_callback_handler"

    def __init__(self, log_path: str = "./logs/", run_name: str = name, run_id: uuid.UUID | None = None, **kwargs) -> None:
        super().__init__(**kwargs)
        self.log_path = log_path
        self.run_name = run_name
        self.run_id = run_id

    def save(self) -> None:
        import os
        from langchain.load.dump import dumps
        if not os.path.exists(self.log_path):
            os.makedirs(self.log_path)
        with open(os.path.join(self.log_path, f"log_{self.run_name}_{self.run_id}.json"), "w", encoding="utf-8") as f:
            str_josin = dumps(self.hierarchy, indent=4)
            f.write(str_josin)