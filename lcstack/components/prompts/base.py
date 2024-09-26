from collections.abc import Sequence
from langchain import hub
from langchain_core.prompts.chat import ChatPromptTemplate
def load_prompt(template):
    if isinstance(template, str):
        return ChatPromptTemplate.from_template(template)
    messages = template
    if isinstance(messages, Sequence):
        # TODO: should be list[tuple], force to tuples with inner lists
        tuple_messages = [tuple(m) for m in messages]
        return ChatPromptTemplate.from_messages(tuple_messages)
    raise ValueError("template must be a string or a list of messages")

def load_hub_prompt(name):
    return hub.pull(name)