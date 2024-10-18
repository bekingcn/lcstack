from abc import abstractmethod
import copy
from typing import Any, Dict, Optional

from langchain_core.runnables import Runnable, RunnableConfig
from langchain_core.tools import BaseTool
from langchain_core.runnables.history import RunnableWithMessageHistory

from lcstack.components.chat_history.base import ChatHistoryFactory
from .models import ChainableRunnables, PromptTemplates, LanguageModels, ComponentType
from .component import Component
from .parsers.base import (
    NamedMappingParserArgs,
    StructOutputParser,
    parse_data_with_struct_mapping,
    OutputParserType,
    parse_data_with_type,
)
from .parsers.mako import eval_expr

CHAT_HISTORY_PARAM_NAME = "chat_history"
DEFAULT_CHAT_HISTORY_KEY = CHAT_HISTORY_PARAM_NAME
NAME_TOOL_SCHEMA = "tool_schema"


class BaseContainer:
    def __init__(
        self, name: str, component: Component, init_kwargs: dict, shared=False
    ):
        self.name = name
        self.component: Component = copy.copy(component)  # shallow copy is enough?
        self.constructor = self.component.func_or_class
        self._kwargs = init_kwargs
        self.params = component.params
        self.component_name = component.name

        # if this runnable is shared and build only once
        self.shared = shared
        # keep a reference to the shared instance
        self.internal = None

    def _build_original(
        self, node_name: Optional[str] = None
    ) -> Any:
        if self.shared and self.internal is not None:
            return self.internal
        kwargs = {}
        params = self.component.params or {}
        for k, v in self._kwargs.items():
            # skip args like "tool_schema" ...
            if k in [NAME_TOOL_SCHEMA]:
                continue
            if isinstance(v, BaseContainer):
                kwargs[k] = v.build_original(node_name=None)
            elif isinstance(v, list):
                kwargs[k] = [
                    x.build_original(node_name=None)
                    if isinstance(x, BaseContainer)
                    else x
                    for x in v
                ]
            elif isinstance(v, dict):
                # TODO: Not supporting tools conversion for dict properties
                kwargs[k] = {
                    k1: v1.build_original(node_name=None)
                    if isinstance(v1, BaseContainer)
                    else v1
                    for k1, v1 in v.items()
                }
            else:
                kwargs[k] = v
        self.internal = self.constructor(**kwargs)
        # :( if prompt template, runnable binding is not supported
        # if node_name and isinstance(self.internal, Runnable):
        #     self.internal = self.internal.with_config(name=node_name)
        return self.internal

    @abstractmethod
    def build_original(
        self, node_name: Optional[str] = None, wrapping_io: bool = False
    ) -> Any:
        """build the callable which could be passed to other components as a reference"""
        raise NotImplementedError(
            f"{self.component_name}: cannot be built as a callable directly in BaseContainer"
        )

    @abstractmethod
    def build(
        self, node_name: Optional[str] = None
    ) -> Any:
        """build the langchain runnable as a running node"""
        raise NotImplementedError(
            f"{self.component_name}: cannot be built as a runnable directly in BaseContainer"
        )

    @abstractmethod
    def invoke(self, inputs: Any, config: Optional[RunnableConfig] = None):
        """
        Invoke the component

        Args:
            input (Any): The input to the component
            config (Optional[RunnableConfig], optional): The configuration for the component. Defaults to None.
        """


class NoneRunnableContainer(BaseContainer):

    def build_original(
        self, node_name: Optional[str] = None, wrapping_io: bool = False
    ) -> Any:
        return super()._build_original(node_name=node_name)
    
    def build(
        self, node_name: Optional[str] = None
    ) -> Any:
        return super()._build_original(node_name=node_name)

    def invoke(self, inputs: Any, config: Optional[RunnableConfig] = None):
        # Could be: DocumentLoader, DocumentTransformer, VectorDB, Embeddings, ChatMessageHistory
        raise NotImplementedError(
            f"{type(self.internal)} from {self.component_name}: cannot be invoked directly in NoneRunnableContainer"
        )


class RunnableContainer(BaseContainer):
    def __init__(
        self,
        name: str,
        component,
        init_kwargs: dict,
        shared=False,
        memory: Optional[ChatHistoryFactory] = None,
        input_mapping: Dict[Optional[str], NamedMappingParserArgs] = None,
        output_mapping: Dict[Optional[str], NamedMappingParserArgs] = None,
        input_expr: Optional[str] = None,
        output_expr: Optional[str] = None,
    ):
        super().__init__(
            name,
            component,
            init_kwargs,
            shared=shared,
        )

        if memory and not isinstance(self.memory, ChatHistoryFactory):
            raise ValueError(
                "The chat history must be an instance of ChatHistoryFactory."
            )
        self.memory = memory
        self.inputs: Dict[Optional[str], NamedMappingParserArgs] = self.component.inputs
        self.default_output_parser_args = self.component.default_output_parser_args

        self.output_mapping = output_mapping
        self.input_mapping = input_mapping
        self.input_expr = input_expr
        self.output_expr = output_expr

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

    def _wrap_memory(self, runnable):
        if isinstance(self.memory, ChatHistoryFactory):
            # TODO: check types which could not be wrapped with message history
            if self.component.component_type in PromptTemplates:
                raise ValueError(
                    "Message history is not supported for prompt templates."
                )
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

            return RunnableWithMessageHistory(
                runnable,
                self.memory.get_by_session_id,
                input_messages_key=input_messages_key,
                output_messages_key=output_messages_key,
                history_messages_key=history_messages_key,
            ).with_config(name=f"{runnable.name}_memory")

        return runnable
    
    def _wrap_tool(self, runnable):
        _runnable = runnable
        if NAME_TOOL_SCHEMA in self._kwargs and not isinstance(_runnable, BaseTool):
            # TODO: better way to config and get the tool schema
            # convert the runnable to a langchain tool
            from langchain_core.tools import convert_runnable_to_tool

            tool_schema = (
                self._kwargs.get(NAME_TOOL_SCHEMA)
                or self.component.tool_schema
                or {}
            )
            if not tool_schema:
                raise UserWarning(
                    f"Tool data not found for {self.component_name} which is used to convert runnable to tool"
                )
            tool_name = tool_schema.get("name")
            tool_desc = tool_schema.get("description")
            tool_arg_types = tool_schema.get("arg_types")
            _runnable = convert_runnable_to_tool(
                runnable=runnable,
                name=tool_name,
                description=tool_desc,
                arg_types=tool_arg_types,
            ).with_config(name=f"{runnable.name}_tool")
            print(
                f"Runnable `{self.component_name}` was converted to a tool:\n",
                runnable,
            )

        return _runnable

    def _original_input_parser(self, input):
        # try expression first, assume only one of input_expr and input_mapping is specified
        if self.input_expr:
            input = eval_expr(self.input_expr, input)
            if input is None:
                raise ValueError(
                    f"Invalid expression for workflow inputs: {self.input_expr}"
                )
        if self.input_mapping:
            input = parse_data_with_struct_mapping(
                data=input, struct_mapping=self.input_mapping
            )
        if not all(
            [
                k == v.name and v.output_type == OutputParserType.pass_through
                for k, v in self.inputs.items()
            ]
        ):
            input_parser = StructOutputParser(
                struct_mapping=self.inputs,
                message_key=None,
                messages_key=None,
                strict=True,
            )
            input = input_parser.parse(input)
        return input

    def _original_output_parser(self, output, node_name: Optional[str] = None):
        if self.default_output_parser_args.output_type != OutputParserType.pass_through:
            args = self.default_output_parser_args.model_dump()
            output_parser_type = args.pop("output_type")
            if not self.default_output_parser_args.message_name:
                args["message_name"] = node_name
            output = parse_data_with_type(
                data=output, output_type=output_parser_type, **args
            )
        # try expression first, assume only one of input_expr and input_mapping is specified
        if self.output_expr:
            output = eval_expr(self.output_expr, output)
            if output is None:
                raise ValueError(
                    f"Invalid expression for workflow inputs: {self.output_expr}"
                )
        if self.output_mapping:
            output = parse_data_with_struct_mapping(
                data=output, struct_mapping=self.output_mapping
            )
        return output

    def _wrap_input_output(self, runnable: Runnable):
        """for now, we add external wrapper (chat history), output parser and tool wrapper"""
        _runnable = runnable
        # TODO: make sure which runnable types should be chained with output parser?
        if self.component.component_type in ChainableRunnables:
            wrap_input = (
                not all(
                    [
                        k == v.name and v.output_type == OutputParserType.pass_through
                        for k, v in self.inputs.items()
                    ]
                )
                or self.input_mapping
                or self.input_expr
            )
            wrap_output = (
                self.default_output_parser_args.output_type
                != OutputParserType.pass_through
                or self.output_mapping
                or self.output_expr
            )
            if wrap_input and wrap_output:
                _runnable = (
                    self._original_input_parser
                    | _runnable
                    | self._original_output_parser
                    # TODO: node name is not used, is necessary to pass it here?
                    # | functools.partial(self._original_output_parser, node_name=node_name)
                )
            elif wrap_input:
                _runnable = self._original_input_parser | _runnable
            elif wrap_output:
                _runnable = (
                    _runnable | self._original_output_parser
                    # TODO: node name is not used, is necessary to pass it here?
                    # | functools.partial(self._original_output_parser, node_name=node_name)
                )
            if _runnable is not runnable:
                _runnable = _runnable.with_config(name=f"{runnable.name}_io")
        return _runnable

    def build_original(
        self, node_name: Optional[str] = None, wrapping_io: bool = False
    ) -> Any:
        node_name = node_name or self.name
        runnable: Runnable = self._build_original(node_name=node_name)
        # make sure the runnable is an instance of Runnable
        if not isinstance(runnable, Runnable):
            raise ValueError(
                f"The internal runnable ({type(runnable)}) is not an instance of Runnable"
            )
        runnable = self._wrap_memory(runnable)
        if wrapping_io and self.component.component_type in ChainableRunnables:
            runnable = self._wrap_input_output(runnable)
        runnable = self._wrap_tool(runnable)
        return runnable

    # TODO: to clarify build() and build_original(), what are their cases?
    def build(self, node_name: Optional[str] = None):
        node_name = node_name or self.name
        runnable = self._build_original(node_name=node_name)
        # make sure the runnable is an instance of Runnable
        if not isinstance(runnable, Runnable):
            raise ValueError(
                f"The internal runnable ({type(runnable)}) is not an instance of Runnable"
            )
        runnable = self._wrap_memory(runnable)
        runnable = self._wrap_input_output(runnable)
        runnable = self._wrap_tool(runnable)
        return runnable

    def invoke(self, inputs: Any, config: Optional[RunnableConfig] = None):
        runnable = self.build()
        return runnable.invoke(inputs, config)


class LLMContainer(RunnableContainer):

    def build(self):
        # TODO: add a wrapper for LLMs
        return super().build()
    
class PromptTemplateContainer(RunnableContainer):

    def build(self):
        # TODO: add a wrapper for PromptTemplates
        return super().build()

class ChainableContainer(RunnableContainer):
    pass

class WorkflowContainer(RunnableContainer):
    pass
