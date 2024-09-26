from fastapi import FastAPI
from langserve import add_routes

from langchain_core.runnables import Runnable
from .core.container import RunnableContainer
from lcstack import LcStackBuilder

# This is a simple implementation of api server using LangServer and FastAPI

# TODO: adapter the chains which contains `kwargs` which is not supported by the current langserve
def start_server(yaml_file: str, components: list[str] = []):
    builder = LcStackBuilder.from_yaml(yaml_file).with_env().with_settings()
    stack = builder.build()

    app = FastAPI(
        title="LangChain Server",
        version="1.0",
        description="A simple api server using Langchain's Runnable interfaces",
    )

    if not components:
        components = [k for k, v in stack.initializers.items()]
    for component in components:
        inst = stack.get(component)
        if isinstance(inst, Runnable):
            # TODO: add chat mode support https://python.langchain.com/v0.1/docs/langserve/#chat-playground
            add_routes(app, inst, path=f"/{component}")
        else:
            raise ValueError(f"{component} is not a runnable")

    import uvicorn
    uvicorn.run(app, host="localhost", port=8000)