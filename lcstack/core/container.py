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
from .parsers.base import MappingParserArgs, NamedMappingParserArgs, PrimitiveOutputParser, StructOutputParser, parse_data_with_struct_mapping

CHAT_HISTORY_PARAM_NAME = "chat_history"
DEFAULT_CHAT_HISTORY_KEY = CHAT_HISTORY_PARAM_NAME
NAME_TOOL_SCHEMA = "tool_schema"

class BaseContainer:
    def __init__(self, 
                 name: str,
                 component: Component,
                 init_kwargs: dict,
                 shared=False):
        self.name = name
        self.component: Component = copy.copy(component)      # shallow copy is enough?
        self.constructor = self.component.func_or_class
        self._kwargs = init_kwargs
        self.params = component.params
        self.component_name = component.name
        
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
                 shared=False,
                 memory: Optional[ChatHistoryFactory]=None, 
                 input_mapping: Dict[Optional[str], NamedMappingParserArgs]=None,
                 output_mapping: Dict[Optional[str], NamedMappingParserArgs]=None):
        super().__init__(name, component, init_kwargs, shared=shared,)
        
        if memory and not isinstance(self.memory, ChatHistoryFactory):
            raise ValueError("The chat history must be an instance of ChatHistoryFactory.")
        self.memory = memory
        self.inputs: Dict[Optional[str], NamedMappingParserArgs] = self.component.inputs
        self.default_output_parser_args = self.component.default_output_parser_args
        
        self.output_mapping = output_mapping
        self.input_mapping = input_mapping
    
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
        # TODO: specify the input/output message keys
        # if len(self.inputs) > 0:
        #     input_messages_key = list(self.inputs.keys())[0]
        # else:
        #     pt_inputs = self._get_template_inputs()
        #     if len(pt_inputs) > 0:
        #         input_messages_key = list(pt_inputs.keys())[0]
# 
        # if self.output_parser_args.output_type == OutputParserType.struct and len(self.output_parser_args.struct_mapping) > 0:
        #     output_messages_key = list(self.output_parser_args.struct_mapping.keys())[0]
        # elif len(self.outputs) > 0:
        #     output_messages_key = list(self.outputs.keys())[0]
        
        def wrap_func(runnable):
            return RunnableWithMessageHistory(
                runnable, 
                memory.get_by_session_id, 
                input_messages_key=input_messages_key, 
                output_messages_key=output_messages_key, 
                history_messages_key=history_messages_key)

        return wrap_func

    def _original_input_parser(self, input):
        if self.input_mapping:
            input = parse_data_with_struct_mapping(
                data=input, 
                struct_mapping=self.input_mapping
            )
        if not all([k == v.name and v.output_type == OutputParserType.pass_through for k, v in self.inputs.items()]):
            input_parser = StructOutputParser(struct_mapping=self.inputs, message_key=None, messages_key=None, strict=True)
            input = input_parser.parse(input)
        return input
        
    def _original_output_parser(self, output, node_name: Optional[str] = None):
        if self.default_output_parser_args.output_type != OutputParserType.pass_through:
            args = self.default_output_parser_args.model_dump()
            output_parser_type = args.pop("output_type")
            if not self.default_output_parser_args.message_name:
                args["message_name"] = node_name
            output = parse_data_with_type(
                data=output, 
                output_type=output_parser_type,
                **args
            )
        if self.output_mapping:
            output = parse_data_with_struct_mapping(
                data=output, 
                struct_mapping=self.output_mapping
            )
        return output

    def _add_wrappers(self, runnable: Runnable, node_name: Optional[str]):
        """for now, we add external wrapper (chat history), output parser and tool wrapper"""
        _runnable = runnable
        if isinstance(self.memory, ChatHistoryFactory):
            _runnable = self._wrap_memory(self.memory)(_runnable)
        # TODO: make sure which runnable types should be chained with output parser?
        if self.component.component_type in ChainableRunnables:
            import functools
            wrap_input = not all([k == v.name and v.output_type == OutputParserType.pass_through for k, v in self.inputs.items()]) or self.input_mapping
            wrap_output = self.default_output_parser_args.output_type != OutputParserType.pass_through or self.output_mapping
            if wrap_input and wrap_output:
                _runnable = (
                    self._original_input_parser
                    | _runnable
                    | self._original_output_parser
                    # TODO: node name is not used, is necessary to pass it here?
                    # | functools.partial(self._original_output_parser, node_name=node_name)
                )
            elif wrap_input:
                _runnable = (
                    self._original_input_parser
                    | _runnable
                )
            elif wrap_output:
                _runnable = (
                    _runnable
                    | self._original_output_parser
                    # TODO: node name is not used, is necessary to pass it here?
                    # | functools.partial(self._original_output_parser, node_name=node_name)
                )
            
            if NAME_TOOL_SCHEMA in self._kwargs and not isinstance(_runnable, BaseTool):
                # TODO: better way to config and get the tool schema
                # convert the runnable to a langchain tool
                from langchain_core.tools import convert_runnable_to_tool
                tool_schema = self._kwargs.get(NAME_TOOL_SCHEMA) or self.component.tool_schema or {}
                if not tool_schema:
                    raise UserWarning(f"Tool data not found for {self.component_name} which is used to convert runnable to tool")
                tool_name = tool_schema.get("name")
                tool_desc = tool_schema.get("description")
                tool_arg_types = tool_schema.get("arg_types")
                _runnable = convert_runnable_to_tool(runnable=runnable, name=tool_name, description=tool_desc, arg_types=tool_arg_types)
                print(f"Runnable `{self.component_name}` was converted to a tool:\n", runnable)
            if _runnable is not runnable:
                _runnable = _runnable.with_config({"name": "wrapped_runnable", "description": "wrapped runnable"})
        return _runnable

    def build_original(self, node_name: Optional[str] = None, as_tool: bool = False) -> Any:
        # avoid casscading as_tool and node_name
        runnable: Runnable = super()._build_original()
        runnable = self._add_wrappers(runnable, node_name)
        return runnable

    # TODO: to clarify build() and build_original(), what are their cases?
    def build(self):
        runnable = self.build_original()
        if not isinstance(runnable, Runnable):
            raise ValueError(f"The internal runnable ({type(runnable)}) is not an instance of Runnable")
        # to avoid confusion, do not covert input/output here any more
        # ecah component should take care of its own input/output        
        return runnable
    
    def invoke(self, inputs: Any, config: Optional[RunnableConfig] = None):
        runnable = self.build()
        return runnable.invoke(inputs, config)

class WorkflowContainer(RunnableContainer):
    pass
