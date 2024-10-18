from typing import Any, Dict
from langchain_core.runnables import Runnable, RunnableLambda, RunnableParallel
import functools

def keyed_value(value: Any, key: str):
    return {key: value}

def dekey_value(value: Dict[str, Any], key: str):
    return value[key]

def filter_out_keys(d: Dict[str, Any], keys: list[str]):
    return {k: v for k, v in d.items() if k not in keys}

def keyed_value_runnable(key: str):
    runnable = RunnableLambda(
        name="keyed_value",
        func=functools.partial(keyed_value, key=key),
    )

    return runnable

def dekey_value_runnable(key: str):
    runnable = RunnableLambda(
        name="dekey_value",
        func=functools.partial(dekey_value, key=key),
    )
    return runnable

def filter_out_keys_runnable(keys: list[str]):
    runnable = RunnableLambda(
        name="filter_out_keys",
        func=functools.partial(filter_out_keys, keys=keys),
    )
    return runnable