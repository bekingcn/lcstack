import copy
from typing import Any, Callable, Optional, Union
from pydantic import BaseModel, Field

from .parsers import OutputParserType, MappingParserArgs, NamedMappingParserArgs, default_output_parser_args_with_type
from .models import ComponentParameter, ComponentType

class Component(BaseModel):
    """
    Represents a component in the stack.
    """
    
    name: str
    """The name of the component."""
    description: str = Field(default="")
    """The description of the component."""
    component_type: ComponentType
    """The type of the component."""
    func_or_class: Callable
    """The function or class to build a instance of target."""
    params: dict[str, ComponentParameter] = Field(default_factory=dict)
    """The parameters to build the runnable or callable."""
    # if dict, the key is the to key, value is the from key, following struct_mapping in the output parser
    inputs: Union[str, list[str],dict[str, Union[str, NamedMappingParserArgs]]] = Field(default_factory=dict)
    """The inputs of the Runnable."""
    tool_schema: Optional[dict[str, Any]] = None
    """The tool schema if it will be converted to a tool."""
    
    supported_output_types: list[OutputParserType] = Field(default=[])
    """The supported output types, not used now and will raise an error on runtime if not supported."""
    default_output_parser_args: MappingParserArgs = default_output_parser_args_with_type(OutputParserType.pass_through)
    """The default output parser args in regesting, which could be overwritten from config stage."""

    def __init__(self, **data):
        super().__init__(**data)

        inputs = self.inputs
        # re-process the inputs
        if isinstance(inputs, str):
            self.inputs = {None: NamedMappingParserArgs(name=inputs, output_type=OutputParserType.pass_through)}
        elif isinstance(inputs, list):
            self.inputs = {v: NamedMappingParserArgs(name=v, output_type=OutputParserType.pass_through) for v in inputs}
        elif isinstance(inputs, dict):
            for k, v in inputs.items():
                if isinstance(v, str):
                    inputs[k] = NamedMappingParserArgs(name=v, output_type=OutputParserType.pass_through)
        # TODO: convert to default_input_parser

class LcComponent(Component):
    """
    Represents a langchain component in the stack.
    """
    inputs: Union[str, list[str],dict[str, Union[str, NamedMappingParserArgs]]] = Field(default_factory=dict)
# support only runnable interface's components, chain/compiled graph/llm/chatmodel/prompt/retriver ...
class RunnableComponent(LcComponent):
    pass

class TodoComponent(RunnableComponent):
    pass