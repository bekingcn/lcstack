from collections.abc import Sequence
from typing import Dict, Any
from langchain import hub
from langchain_core.prompts.chat import ChatPromptTemplate, MessageLike
from langchain_core.prompt_values import PromptValue
from langchain_core.runnables import Runnable

def load_prompt_from_messages(messages: Sequence):
    # TODO: should be list[str | tuple], force to tuples with inner lists
    tuple_messages = []
    for m in messages:
        if isinstance(m, str):
            tuple_messages.append(("human", m))
        elif isinstance(m, tuple) and len(m) == 2:
            tuple_messages.append(m)
        elif isinstance(m, MessageLike):
            tuple_messages.append(m)
        elif isinstance(m, list):
            tuple_messages.append(tuple(m))
        else:
            raise ValueError(f"Invalid message format: {m}")

    return ChatPromptTemplate.from_messages(tuple_messages)

def load_prompt(template):
    if isinstance(template, str):
        return ChatPromptTemplate.from_template(template)
    if isinstance(template, Sequence):
        return load_prompt_from_messages(messages=template)
    raise ValueError("template must be a string or a list of messages, strings, or lists")


def load_hub_prompt(name):
    return hub.pull(name)

def create_prompt_node(template: ChatPromptTemplate, output_key="messages") -> Runnable:

    def to_messages(prompt_value: PromptValue) -> Dict[str, Any]:
        return {output_key: prompt_value.to_messages()}
    return (
        template
        | to_messages
    )