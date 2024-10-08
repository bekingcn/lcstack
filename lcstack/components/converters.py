from langchain_core.runnables import Runnable, RunnableLambda

from ..core.parsers import OutputParserType, parse_data_with_type


def create_data_parser(
    output_type: OutputParserType,
    input_type: OutputParserType = OutputParserType.pass_through,
    **kwargs,
) -> Runnable:
    return RunnableLambda(
        lambda data: parse_data_with_type(data, output_type, input_type, **kwargs)
    ).with_config({"run_name": "data_parser"})


# TODO: process data with Jinja template
