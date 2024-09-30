from langchain_core.output_parsers import (
    BaseOutputParser,

    StrOutputParser,
    SimpleJsonOutputParser,
    JsonOutputParser,
    
    CommaSeparatedListOutputParser,
    MarkdownListOutputParser,
    NumberedListOutputParser,

    JsonOutputToolsParser,
    PydanticToolsParser,
)

from lcstack.core.parsers.base import BaseOutputParser

# NOTE: this is LLM/Chat Model's output parser, 
# it's different from the chain's output parsers under core/parsers

def create_output_parser(type: str, name: str|None = None, **kwargs) -> BaseOutputParser:
    """Create an langchain output parser with the given type.

    Args:
        type: The type of the output parser. 
        Supported types are 
        "str", "json", "comma_separated_list", "markdown_list", "numbered_list", "json_tools", "pydantic_tools".

    Returns:
        The created output parser.
    """

    if type == "str":
        return StrOutputParser(name=name)
    elif type == "json":
        return JsonOutputParser(name=name, **kwargs)
    elif type == "comma_separated_list":
        return CommaSeparatedListOutputParser(name=name)
    elif type == "markdown_list":
        return MarkdownListOutputParser(name=name, **kwargs)
    elif type == "numbered_list":
        return NumberedListOutputParser(name=name, **kwargs)
    elif type == "json_tools":
        return JsonOutputToolsParser(name=name, **kwargs)
    elif type == "pydantic_tools":
        return PydanticToolsParser(name=name, **kwargs)
    else:
        raise ValueError(f"Invalid output parser type: {type}")