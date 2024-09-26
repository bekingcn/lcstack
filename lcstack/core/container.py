from abc import abstractmethod
import copy
from pydantic import BaseModel
from typing import Any, Callable, Dict, Optional

from langchain_core.runnables import Runnable, RunnableConfig
from langchain_core.tools import BaseTool
from langchain_core.runnables.history import RunnableWithMessageHistory

from lcstack.components.chat_history.base import ChatHistoryFactory
from .models import ChainableRunnables, ComponentType
from .component import Component
from .parsers.base import MappingParserArgs, NamedMappingParserArgs, PrimitiveOutputParser, StructOutputParser

CHAT_HISTORY_PARAM_NAME = "chat_history"
DEFAULT_CHAT_HISTORY_KEY = CHAT_HISTORY_PARAM_NAME
NAME_TOOL_SCHEMA = "tool_schema"

class BaseContainer:
    def __init__(self, 
                 name: str,
                 component: Component,
                 init_kwargs: dict, 
                 outputs: dict[str, NamedMappingParserArgs], 
                 shared=False):
        self.name = name
        self.component: Component = copy.copy(component)      # shallow copy is enough?
        self.constructor = self.component.func_or_class
        self._kwargs = init_kwargs
        self.params = component.params
        self.component_name = component.name
        self.inputs = self.component.inputs
        self.outputs = outputs
        
        # if this runnable is shared and build only once
        self.shared = shared
        # keep a reference to the shared instance
        self.internal = None

    def _build_original(self, node_name: Optional[str] = None, as_tool: bool = False) -> Any:
        if self.shared and self.internal is not None:
            return self.internal
        kwargs = {}
        params = self.component.params or {}
        for k, v in self._kwargs.items():
            # skip args like "tool_schema" ...
            if k in [NAME_TOOL_SCHEMA]:
                continue
            # any other types should be handled from the config supports?
            as_tool = (
                k in params and 
                params[k].component_type == ComponentType.Tool
            )
            if isinstance(v, BaseContainer):
                kwargs[k] = v.build_original(node_name=node_name, as_tool=as_tool)
            elif isinstance(v, list):
                kwargs[k] = [x.build_original(node_name=node_name, as_tool=as_tool) if isinstance(x, BaseContainer) else x for x in v]
            elif isinstance(v, dict):
                # TODO: Not supporting tools conversion for dict properties
                kwargs[k] = {k1: v1.build_original(node_name=node_name) if isinstance(v1, BaseContainer) else v1 for k1, v1 in v.items()}
            else:
                kwargs[k] = v
        self.internal = self.constructor(**kwargs)
        return self.internal
    
    @abstractmethod
    def build_original(self, node_name: Optional[str] = None, as_tool: bool = False) -> Any:
        """build the langchain runnables oe callables which could be passed to other components"""
        raise NotImplementedError(f"{self.component_name}: cannot be built directly in BaseContainer")
    
    @abstractmethod
    def invoke(self, input: Any, config: Optional[RunnableConfig] = None):
        """
        Invoke the component

        Args:
            input (Any): The input to the component
            config (Optional[RunnableConfig], optional): The configuration for the component. Defaults to None.
        """

class NoneRunnableContainer(BaseContainer):
    def invoke(self, input: Any, config: Optional[RunnableConfig] = None):
        # Could be: DocumentLoader, DocumentTransformer, VectorDB, Embeddings, ChatMessageHistory
        raise NotImplementedError(f"{type(self.internal)} from {self.component_name}: cannot be invoked directly in NoneRunnableContainer")
    
    def build_original(self, node_name: Optional[str] = None, as_tool: bool = False) -> Any:
        return super()._build_original(node_name=node_name, as_tool=as_tool)

from .parsers import OutputParserType, parse_data_with_type
class RunnableContainer(BaseContainer):
    def __init__(self, 
                 name: str,
                 component, 
                 init_kwargs: dict, 
                 outputs: dict[str, NamedMappingParserArgs], 
                 shared=False,
                 memory: Optional[ChatHistoryFactory]=None, 
                 output_parser_args: Optional[MappingParserArgs]=None):
        super().__init__(name, component, init_kwargs, outputs, shared=shared,)
        
        if memory and not isinstance(self.memory, ChatHistoryFactory):
            raise ValueError("The chat history must be an instance of ChatHistoryFactory.")
        self.memory = memory
        self.output_parser_args = output_parser_args
        self.output_parser_type = output_parser_args.output_type
    
    def _get_template_inputs(self) -> Dict[str, str]:
        # trying get the inputs from prompt templates        
        from langchain_core.prompts import BasePromptTemplate
        inputs = {}
        for pt in self._kwargs.values():
            if isinstance(pt, BasePromptTemplate):
                for iv in pt.input_variables:
                    if iv not in inputs:
                        inputs[iv] = iv
        return inputs

    def _wrap_memory(self, memory):
        # TODO: improve this with more input or output info
        input_messages_key = None
        output_messages_key = None
        # TODO: for now, fixed the key name
        history_messages_key = DEFAULT_CHAT_HISTORY_KEY
        
        # find the first input or output as the keys
        if len(self.inputs) > 0:
            input_messages_key = list(self.inputs.keys())[0]
        else:
            pt_inputs = self._get_template_inputs()
            if len(pt_inputs) > 0:
                input_messages_key = list(pt_inputs.keys())[0]

        if self.output_parser_args.output_type == OutputParserType.struct and len(self.output_parser_args.struct_mapping) > 0:
            output_messages_key = list(self.output_parser_args.struct_mapping.keys())[0]
        elif len(self.outputs) > 0:
            output_messages_key = list(self.outputs.keys())[0]
        
        def wrap_func(runnable):
            return RunnableWithMessageHistory(
                runnable, 
                memory.get_by_session_id, 
                input_messages_key=input_messages_key, 
                output_messages_key=output_messages_key, 
                history_messages_key=history_messages_key)

        return wrap_func

    def _original_output_parser(self, output, node_name: Optional[str] = None):
        if not self.output_parser_args:
            return output
        output_parser_args = self.output_parser_args.model_dump()
        output_parser_type = output_parser_args.pop("output_type")
        if not self.output_parser_args.message_name:
            output_parser_args["message_name"] = node_name
        return parse_data_with_type(
            data=output, 
            output_type=output_parser_type,
            **output_parser_args
        )

    def _add_wrappers(self, runnable: Runnable, node_name: Optional[str]):
        """for now, we add external wrapper (chat history), output parser and tool wrapper"""
        if isinstance(self.memory, ChatHistoryFactory):
            runnable = self._wrap_memory(self.memory)(runnable)

        # TODO: make sure which runnable types should be chained with output parser?
        if self.component.component_type in ChainableRunnables:
            # assuming all inputs's values `NamedMappingParserArgs`
            pass_through = all([k == v.name and v.output_type == OutputParserType.pass_through for k, v in self.inputs.items()])
            if not pass_through:
                input_parser = StructOutputParser(struct_mapping=self.inputs)
                runnable = input_parser.parse | runnable
                # otherwise, the inputs are passed through
            
            if self.output_parser_type != OutputParserType.pass_through:
                import functools
                runnable = (
                    runnable 
                    | functools.partial(self._original_output_parser, node_name=node_name)
                )
            if NAME_TOOL_SCHEMA in self._kwargs and not isinstance(runnable, BaseTool):
                # convert the runnable to a langchain tool
                from langchain_core.tools import convert_runnable_to_tool
                tool_schema = self._kwargs.get(NAME_TOOL_SCHEMA) or self.component.tool_schema or {}
                if not tool_schema:
                    raise UserWarning(f"Tool data not found for {self.component_name} which is used to convert runnable to tool")
                tool_name = tool_schema.get("name")
                tool_desc = tool_schema.get("description")
                tool_arg_types = tool_schema.get("arg_types")
                runnable = convert_runnable_to_tool(runnable=runnable, name=tool_name, description=tool_desc, arg_types=tool_arg_types)
                print(f"Runnable `{self.component_name}` was converted to a tool:\n", runnable)
        return runnable

    def build_original(self, node_name: Optional[str] = None, as_tool: bool = False) -> Any:
        # avoid casscading as_tool and node_name
        runnable: Runnable = super()._build_original()
        runnable = self._add_wrappers(runnable, node_name)
        return runnable

    def build(self):
        runnable = self.build_original()
        if not isinstance(runnable, Runnable):
            raise ValueError(f"The internal runnable ({type(runnable)}) is not an instance of Runnable")
        # to avoid confusion, do not covert input/output here any more
        # ecah component should take care of its own input/output        
        return runnable
        # return (
        #     self._default_enter_func
        #     | runnable
        #     | self._default_exit_func
        # )
    
    def invoke(self, inputs: Any, config: Optional[RunnableConfig] = None):
        runnable = self.build()
        return runnable.invoke(inputs, config)
    
    # TODO: remove, or add before _build_original for checking inputs
    # add this to build_original? hard to do now
    # it's dependent on the chians/agents, and invariables of prompt templates
    def _default_enter_func(self, input_value):
        # NOTE: supported input types: str, dict[str, Any]
        input_keys = list(self.inputs.keys())
        if isinstance(input_value, dict):
            missing_keys = set(input_keys).difference(input_value)
            if len(missing_keys) > 1:   # chat history 
                raise KeyError(f"Missing keys {missing_keys} in input.")
        elif isinstance(input_value, str):
            if len(input_keys) == 1:
                input_value = {input_keys[0]: input_value}
            elif len(input_keys) > 1:
                raise ValueError("Only one input is supported for non-dict inputs.")
        else:
            raise ValueError(f"Unsupported input type: {type(input_value)}")

        return input_value
    
    # TODO: remove, to unify with _original_output_parser
    def _default_exit_func(self, output_value):
        # Try to convert the output to a string if no output_keys or single output
        from langchain_core.messages import BaseMessage
        output_keys = list(self.outputs.keys())
        if isinstance(output_value, BaseMessage):
            return output_value.content
        elif isinstance(output_value, dict):
            if len(output_keys) == 0:
                return output_value
                # return output_value.get(output_keys[0])
            else:
                missing_keys = set(output_keys).difference(output_value)
                if len(missing_keys) > 0:
                    raise KeyError(f"Missing keys {missing_keys} in output.")
                return {key: value for key, value in output_value.items() if key in output_keys}
        return output_value

class WorkflowContainer(RunnableContainer):
    pass
