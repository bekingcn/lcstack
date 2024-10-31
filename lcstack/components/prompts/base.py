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
        elif isinstance(m, list) and len(m) == 2:
            tuple_messages.append(tuple(m))
        elif isinstance(m, tuple) and len(m) == 2:
            tuple_messages.append(m)
        elif isinstance(m, MessageLike):
            tuple_messages.append(m)
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

def create_prompt_node(
        template: ChatPromptTemplate | Sequence | str, 
        output_key="messages", 
        # PRIVATE: used for `build_original` in `BaseContainer` for reference as a prompt template
        return_oreiginal=False
    ) -> Runnable:

    if not isinstance(template, ChatPromptTemplate):
        template = load_prompt(template)

    if return_oreiginal:
        return template
    def to_messages(prompt_value: PromptValue) -> Dict[str, Any]:
        return {output_key: prompt_value.to_messages()}
    return (
        template
        | to_messages
    ).with_config(name="prompt_node")