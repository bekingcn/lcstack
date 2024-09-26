from langchain_core.runnables import Runnable, RunnableLambda

from ..core.parsers import OutputParserType, parse_data_with_type

def create_data_parser(input_type: OutputParserType, output_type: OutputParserType, **kwargs) -> Runnable:
    
    return RunnableLambda(lambda data: parse_data_with_type(data, output_type, input_type, **kwargs))

# TODO: process data with Jinja template