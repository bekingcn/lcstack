from logging import warning
from copy import deepcopy
from enum import Enum
from abc import abstractmethod
import operator
from typing import List, Dict, Any, Annotated, Callable, Optional, TypedDict, Union
from pydantic import computed_field, BaseModel, Field

from langchain_core.messages import BaseMessage
from langchain_core.runnables import Runnable, RunnableLambda
from langgraph.graph import START, END, StateGraph
from langgraph.checkpoint.memory import MemorySaver

from lcstack.core.parsers import NamedMappingParserArgs
from lcstack.registry import get_component

from ..core.parsers import DataType, parse_data_with_type, default_value_with_type
from ..core.container import RunnableContainer

NAME_BRANCH_STEPS = "branch__steps__"

# TODO: align with DataType
class SupportedDataType(str, Enum):
    str = "str"
    int = "int"
    float = "float"
    bool = "bool"
    
    struct = "struct"
    list = "list"
    message = "message"
    messages = "messages"
    
    any = "any"
    structlist = "structlist"

supported_types_map = {
    SupportedDataType.str: str,
    SupportedDataType.int: int,
    SupportedDataType.float: float,
    SupportedDataType.bool: bool,
    SupportedDataType.struct: dict,
    SupportedDataType.list: list,
    SupportedDataType.message: BaseMessage,
    SupportedDataType.messages: List[BaseMessage],

    SupportedDataType.any: Any,
    
    SupportedDataType.structlist: List[Dict],
}

class SupportedOperators(str, Enum):
    Add = "add"
    Default = "default"

class FieldConfig(BaseModel):
    name: str
    field_type: SupportedDataType = SupportedDataType.str
    description: Optional[str] = ""
    default: Optional[Any] = None

    operator: Optional[SupportedOperators] = SupportedOperators.Default
    required: bool = Field(default=False)
    converter: Optional[Callable[..., Any]] = None

class AgentConfig(BaseModel):
    config: str
    """config file path, `[config].yaml` or `[config].yaml:[agent name]`"""
    name: str
    """name of the agent, will be parsed from `config` if not specified"""
    kwargs: Dict[str, Any] = Field(default_factory=dict)


class BaseVertex(BaseModel):
    name: str
    # TODO: assume the `next` is always useful, right?
    next: Optional[str] = None

class CallableVertex(BaseVertex):
    # TODO: Any is not good, any better to represent RunnableContainer?
    agent: Union[str, AgentConfig, Callable, Any]    # Any: RunnableContainer
    """agent config file path, `[config].yaml:[agent name]`, config file path and name, or parsed agent"""
    # key is schema field name, value is input or output name
    # for input_mapping, value is the input name
    # if input_mapping is not dict, it will pop the key as input (if is MappingParserArgs, will be converted to output type)
    input_mapping: Union[str, NamedMappingParserArgs, Dict[Optional[str], Union[str, NamedMappingParserArgs]]] = Field(default_factory=dict)
    output_mapping: Dict[str, Union[str, NamedMappingParserArgs]] = Field(default_factory=dict)

    def __init__(self, **data):
        super().__init__(**data)

        if isinstance(self.agent, str):
            config, _, name = self.agent.partition(":")
            if not config or not name.strip():
                raise ValueError(f"Invalid agent config `{self.agent}`, should be `config_file:agent_name`")
            self.agent = AgentConfig(config=config.strip(), name=name.strip())

        if isinstance(self.input_mapping, str):
            self.input_mapping = {None: NamedMappingParserArgs(name=self.input_mapping, output_type=DataType.pass_through)}
        elif isinstance(self.input_mapping, NamedMappingParserArgs):
            self.input_mapping = {None: self.input_mapping}


class BaseWorkflowModel(BaseModel):
    name: Optional[str]
    vertices: List[BaseVertex] = Field(default_factory=list[BaseVertex])
    # TODO: resolve: UserWarning: Field name "schema" shadows an attribute in parent "BaseModel
    schema_: List[FieldConfig] = Field(default_factory=list[FieldConfig], alias="schema")
    # map schema (state) name to input name
    input_mapping: Optional[Dict[str, str]] = {}
    reset_state: Optional[bool] = Field(default=False)

## helper functions

def _default_value_with_type(_type: SupportedDataType) -> Any:
    if _type == SupportedDataType.str:
        return ""
    elif _type == SupportedDataType.int:
        return 0
    elif _type == SupportedDataType.float:
        return 0.0
    elif _type == SupportedDataType.bool:
        return False
    else:
        return default_value_with_type(_type)
def to_parsed_type(data_type: SupportedDataType) -> DataType:    
    _type = (
        DataType.primitive
        if data_type in [SupportedDataType.str, SupportedDataType.int, SupportedDataType.float, SupportedDataType.bool]
        else DataType.struct if data_type == SupportedDataType.struct
        else DataType.list if data_type == SupportedDataType.list
        else DataType.message if data_type == SupportedDataType.message
        else DataType.messages if data_type == SupportedDataType.messages
        else DataType.pass_through
    )
    return _type

# NOTE: we are using TypedDict to create a model. is it better to use BaseModel?
def typed_dict_create_model(model_name, fields: List[FieldConfig]):

    new_fields = {}
    for f in fields:
        _type = supported_types_map[f.field_type]
        field_name = f.name
        op = f.operator
        if op and op == SupportedOperators.Add:
            # TODO: add for string, hopefuly with line breaks
            if _type == str:
                _type = Annotated[str, lambda a, b: (a + "\n\n" + b).strip("\n")]
            elif _type == BaseMessage:
                # TODO: Message, to a list of Messages?
                _type = Annotated[List[BaseMessage], operator.add]
            else:
                # other types: List, Dict, Messages
                _type = Annotated[_type, operator.add]
        new_fields[field_name] = _type
    return TypedDict(model_name, new_fields)

class Workflow:
    def __init__(self, model: BaseWorkflowModel):
        self.name: str = model.name or self.__class__.__name__
        self.vertices: List[BaseVertex] = model.vertices
        self.input_mapping = model.input_mapping or {}
        self.schema = model.schema_ or []
        self.reset_state = model.reset_state
        
        self.schema_fields_map = { f.name: f for f in self.schema }
        self._model = model
        self._default_dict = {}
        self.start_nodes: List[BaseVertex] = []

        self.graph = None
        self.intermediate_steps = []

        self.checkpoint = None
        self.last_thread_id = None

    graph: StateGraph = None

    @computed_field
    def state_model(self):
        return self.build_state_model()

    def build_state_model(self):
        """This collects all the fields from graph's schema.
        
        TODO: a better way is to collect all the fields from chain/agent nodes' output and input types.
            with edges definations like: node1.output.field1 -> node2.input.field2"""
        fields = self.schema.copy()
        # add a extra `branch_steps` field with special field type
        fields.append(FieldConfig(**{"name": NAME_BRANCH_STEPS, "field_type": SupportedDataType.structlist, "operator": SupportedOperators.Add}))
        
        # camel case the name
        model_name = f"{self.name}State"
        
        DynamicStateModel = typed_dict_create_model(model_name, fields=fields)
        # assign the default values and state fields
        self._default_dict = {f.name: f.default or _default_value_with_type(f.field_type) for f in self.schema}
        
        return DynamicStateModel
    
    @abstractmethod
    def build_graph(self):
        """
        Build the graph from the vertices, edges, and input/output mappings
        """
    
    # TODO: `build` or `compile`?
    def compile(self):
        self.build_graph()
        
        # NOTE: here the checkpoint is used to keep the state each turn to enter the workflow
        #   because we expect the state to be cleared after each turn
        #   If you hope to persist the state, the solution is to wrap another state saver, which sync the state from this checkpoint
        self.checkpoint = MemorySaver()
        workflow = self.graph.compile(checkpointer=self.checkpoint)
        return workflow
    
    def _enter_graph(self, inputs, config=None):
        """_summary_

        Args:
            inputs (_type_): inputs from invoke, a Dict or plain value

        Raises:
            ValueError: ...

        Returns:
            _type_: Dict, to fill the graph state
        """
        
        # NOTE: clear the state, this works with only MemorySaver 
        #   cause we created a MemorySaver with only one thread for each workflow
        #   or, you should clear the state for specific thread manually
        thread_id = (config or {}).get("configurable", {}).get("thread_id", None)
        
        if self.reset_state:
            warning(f"reset state in workflow `{self.name}` for thread `{thread_id}`")
            self.checkpoint.storage.clear()
        
        outputs = {}
        # convert mapping to dict[str, str]
        input_mapping = {sn: inn.name if isinstance(inn, NamedMappingParserArgs) else inn for sn, inn in self.input_mapping.items()}
        start_vertex_name = self.start_nodes[0].name
        
        # all required fields should be in the inputs
        try:
            required_keys = {input_mapping[f.name] for f in self.schema if f.required}
            missing_keys = required_keys.difference(input_mapping.values())
            if len(missing_keys) > 0:
                raise ValueError(f"required fields {missing_keys} are missing in workflow {self.name}'s inputs mapping")
        except KeyError:
            raise ValueError(f"input mapping should include all required fields")

        # force convert the inputs to dict
        # TODO: shuold do this?
        
        optional_key = list(required_keys)[0] if required_keys else list(input_mapping.values())[0] if len(input_mapping) > 0 else self.schema[0].name
        if isinstance(inputs, str):
            warning(f"input is a string, converting it to a dict with key {required_keys[0]}")
            inputs = {optional_key: inputs}
        if isinstance(inputs, dict):
            # process unaligned inputs with one key, mapping to the required key or the first one
            if len(inputs) == 1 and len(required_keys) <= 1:
                actual_key, val = list(inputs.items())[0]
                if actual_key != optional_key and actual_key not in list(input_mapping.values()):
                    warning(f"input key `{actual_key}`is different from required key `{optional_key}`, converting it to `{optional_key}`")
                    inputs = {optional_key: val}
            missing_keys = required_keys.difference(inputs.keys())
            if len(missing_keys) > 0:
                raise ValueError(f"required fields {missing_keys} are missing in workflow {self.name}'s inputs")
        else:
            raise ValueError(f"inputs are not a dict or a plain value, failed to be aligned with required keys")
        
        for f in self.schema:
            sname = f.name
            iname = self.input_mapping.get(sname, None)
            if not iname:
                # fill with default values
                outputs[sname] = self._default_dict.get(sname, None)
            elif isinstance(iname, str) and iname in inputs:
                outputs[sname] = (
                    parse_data_with_type(
                        inputs[iname], 
                        to_parsed_type(f.field_type), 
                        message_role="human",
                        message_name="input"
                    )
                    if not f.converter
                    else f.converter(inputs[iname])
                )
            elif isinstance(iname, NamedMappingParserArgs) and iname.name in inputs:
                # support with NamedMappingParserArgs
                args = iname.model_dump()
                output_name = args.pop("name")
                outputs[sname] = parse_data_with_type(
                    inputs[output_name],
                    to_parsed_type(f.field_type),
                    **args
                )
            else:
                raise ValueError(f"Unsupported input mapping: {iname}")

        return outputs # {**inputs, **outputs}

    def _enter_node_func(self, node_name: str, input_mapping: Dict):
        def _func(state):
            # save a copy of the state
            state_snapshot = deepcopy(state)
            self.intermediate_steps.append((node_name, state_snapshot, ))

            state = parse_data_with_type(
                state,
                output_type=DataType.struct,
                known_data_type=DataType.struct,
                struct_mapping=input_mapping,
            )
            # state name -> input name
            # for sname, iname in input_mapping.items():
            #     # tricky case: if only one mapping and sname is None, return the keyed value by `iname`
            #     if (sname is None or sname == "__") and len(input_mapping) == 1:
            #         return state[iname]
            #     # should not happen here if input_mapping is checked
            #     if sname not in state:
            #         raise ValueError(f"Required state {sname} is missing")
            #     if isinstance(iname, str):
            #         # re-map the state to the node input
            #         if sname in state and iname != sname:
            #             state[iname] = state[sname]
            #     elif isinstance(iname, NamedMappingParserArgs):
            #         # support with NamedMappingParserArgs
            #         args = iname.model_dump()
            #         field = self.schema_fields_map[sname]
            #         output_name = args.pop("name")
            #         output_type=args.pop("output_type"),
            #         state[iname] = parse_data_with_type(
            #             state[output_name],
            #             output_type=output_type,
            #             known_data_type=to_parsed_type(field.field_type),
            #             **args
            #         )
            #     else:
            #         raise ValueError(f"Unsupported input mapping: {iname}")
            return state

        return _func

    def _map_output_func(self, output_mapping: Dict, node_name: str):
        # map outputs to the state
        def _func(outputs):
            ret = {}
            # convert the output to a dict
            if isinstance(outputs, (str, BaseMessage)):
                warning(f"output is a string or BaseMessage, expected a dict. Forcing it to be a dict.")
                # TODO: multiple state names mapping to this output? any side effects?
                new_outputs = {}
                for oname in output_mapping.values():
                    if isinstance(oname, str):
                        new_outputs = {oname: outputs}
                    elif isinstance(oname, NamedMappingParserArgs):
                        new_outputs = {oname.name: outputs}
                outputs = new_outputs
                
            if isinstance(outputs, dict):
                # default mapping
                if not output_mapping:
                    # add mapping, in case of type mismatch
                    for name in outputs:
                        output_mapping[name] = name

                # support multiple state names mapping with one output
                new_output_mapping: Dict[str, NamedMappingParserArgs] = {}
                for sname, oname in output_mapping.items():
                    field = self.schema_fields_map.get(sname, None)
                    if not field:
                        raise ValueError(f"Unsupported state name for output mapping: {sname}")
                    if isinstance(oname, str):
                        new_output_mapping[sname] = NamedMappingParserArgs(
                            name=oname, 
                            output_type=to_parsed_type(field.field_type),
                            message_role="ai",
                            message_name=node_name,
                        )
                        # TODO: process the given converter function 
                        if field.converter:
                            pass
                    elif isinstance(oname, NamedMappingParserArgs):                        
                        new_output_mapping[sname] = oname
                    else:
                        raise ValueError(f"Unsupported output mapping: {oname}")
                
                ret = parse_data_with_type(
                    outputs, 
                    output_type=DataType.struct, 
                    struct_mapping=new_output_mapping, 
                    known_data_type=DataType.struct
                )
            else:
                raise ValueError(f"Unsupported output type: {type(outputs)}")
            return ret

        return _func
    
    def _to_graph_node_name(self, name):
        if name in [START, END]:
            return name
        return f"{self.name}_{name}"
        
    def _build_callable_node(self, v: CallableVertex):
        node_name = self._to_graph_node_name(v.name)
        if isinstance(v.agent, Callable):
            callable = v.agent
        # elif isinstance(v.agent, AgentConfig):
        #     callable = AgentInvoker(node_name=node_name, agent_config=v.agent).runnable
        elif isinstance(v.agent, RunnableContainer):
            callable = v.agent.build_original()
        else:
            raise ValueError(f"Unsupported agent type `{type(v.agent)}` in callable vertex `{v.name}`")
        output_mapping = v.output_mapping or {}
        runnable = (
            RunnableLambda(self._enter_node_func(node_name, input_mapping=v.input_mapping), name=node_name)
            | callable
            | self._map_output_func(output_mapping, node_name=node_name)
        )
        return node_name, runnable
    
    # invoke and stream , which are called directly by the user

    def _gen_thread_id(self):
        if self.last_thread_id is None:
            self.last_thread_id = 1
        self.last_thread_id += 1
        return f"{self.name}_thread_{self.last_thread_id}"
    
    def get_last_thread_id(self) -> str:
        return f"{self.name}_thread_{self.last_thread_id}"

    def invoke(self, inputs, config=None, **kwargs):
        workflow = self.compile()
        config = config or {}
        config["thread_id"] = self._gen_thread_id()
        runnable = (
            self._enter_graph 
            | workflow
        )
        return runnable.invoke(inputs, config, **kwargs)
    
    def stream(self, inputs, config=None, **kwargs):
        workflow = self.compile()
        # stream_mode = config.pop("stream_mode", "values")
        # workflow.stream_mode = stream_mode
        config = config or {}
        config["thread_id"] = self._gen_thread_id()
        runnable = (
            self._enter_graph 
            | workflow
        )
        yield from runnable.stream(inputs, config, **kwargs)