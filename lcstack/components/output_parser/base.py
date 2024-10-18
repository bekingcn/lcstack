from langchain_core.output_parsers import (
    BaseOutputParser,
    StrOutputParser,
    JsonOutputParser,
    CommaSeparatedListOutputParser,
    MarkdownListOutputParser,
    NumberedListOutputParser,
    JsonOutputToolsParser,
    PydanticToolsParser,
)
from ..utils import keyed_value_runnable, dekey_value_runnable

# NOTE: this is LLM/Chat Model's output parser,
# it's different from the chain's output parsers under core/parsers


def create_output_parser(
    type: str, name: str | None = None, input_key: str | None = None, output_key: str | None = None, **kwargs
) -> BaseOutputParser:
    """Create an langchain output parser with the given type.

    Args:
        type: The type of the output parser.
        Supported types are
        "str", "json", "comma_separated_list", "markdown_list", "numbered_list", "json_tools", "pydantic_tools".

    Returns:
        The created output parser.
    """

    if type == "str":
        parser = StrOutputParser(name=name)
    elif type == "json":
        parser = JsonOutputParser(name=name, **kwargs)
    elif type == "comma_separated_list":
        parser = CommaSeparatedListOutputParser(name=name)
    elif type == "markdown_list":
        parser = MarkdownListOutputParser(name=name, **kwargs)
    elif type == "numbered_list":
        parser = NumberedListOutputParser(name=name, **kwargs)
    elif type == "json_tools":
        parser = JsonOutputToolsParser(name=name, **kwargs)
    elif type == "pydantic_tools":
        parser = PydanticToolsParser(name=name, **kwargs)
    else:
        raise ValueError(f"Invalid output parser type: {type}")

    if input_key and output_key:
        runnable = dekey_value_runnable(key=input_key) | parser | keyed_value_runnable(key=output_key)
    elif input_key:
        runnable = dekey_value_runnable(key=input_key) | parser
    elif output_key:
        runnable = parser | keyed_value_runnable(key=output_key)
    else:
        runnable = parser
    return runnable

SUPPORTED_OUTPUT_PARSERS = [
    "str",
    "json",
    "comma_separated_list",
    "markdown_list",
    "numbered_list",
    "json_tools",
    "pydantic_tools",
]