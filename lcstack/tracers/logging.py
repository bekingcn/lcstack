
import uuid
import os, logging
from langchain_core.tracers.stdout import FunctionCallbackHandler

class LoggingCallbackHandler(FunctionCallbackHandler):
    """Tracer that prints to the console."""

    name: str = "logging_callback_handler"

    def __init__(self, log_path: str = "./logs/", run_name: str = name, run_id: uuid.UUID | None = None, **kwargs) -> None:
        if not os.path.exists(log_path):
            os.makedirs(log_path)
        if run_id is None:
            run_id = uuid.uuid4()
        file_name = os.path.join(log_path, f"log_{run_name}_{run_id}.log")
        # if os.path.exists(file_name):
        #     os.remove(file_name)
        # TODO: any good idea to do this?
        # create a file handler
        handler = logging.FileHandler(file_name)
        handler.setLevel(logging.INFO)
        # create a logging format
        formatter = logging.Formatter('%(asctime)s - %(message)s')
        handler.setFormatter(formatter)
        # add the handlers to the logger
        def logger_func(msg):
            handler.emit(logging.LogRecord("", logging.INFO, "", 0, msg, None, None))
        super().__init__(function=logger_func, **kwargs)
        self.file_handler = handler

    def close(self):
        self.file_handler.close()