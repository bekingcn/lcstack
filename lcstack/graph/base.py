from logging import warning
from copy import deepcopy
from enum import Enum
from abc import abstractmethod
import operator
from typing import List, Dict, Any, Annotated, Callable, Optional, TypedDict, Union
from pydantic import computed_field, BaseModel, Field

from langchain_core.messages import BaseMessage
from langchain_core.runnables import RunnableLambda
from langgraph.graph import START, END, StateGraph
from langgraph.checkpoint.memory import MemorySaver

from ..core.parsers import (
    DataType,
    parse_data_with_type,
    default_value_with_type,
    rebuild_struct_mapping,
    NamedMappingParserArgs,
    parse_data_with_struct_mapping,
)
from ..core.container import RunnableContainer
from ..core.parsers.mako import eval_expr
from ..configs import get_expr_enabled

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
    # TODO: Any is not good, any better to represent RunnableContainer which is not BaseModel?
    agent: Union[str, AgentConfig, Callable, Any]  # Any: RunnableContainer
    """agent config file path, `[config].yaml:[agent name]`, config file path and name, or parsed agent"""
    # key is schema field name, value is input or output name
    # convertion: struct -> any
    # str, pop from input as node input
    # dict key:
    #   - None or _, as above, pop from input as node input
    # dict value:
    #   - str, input field name, align to state field type
    #   - NamedMappingParserArgs, convert following args
    input_mapping: Union[
        str,
        NamedMappingParserArgs,
        Dict[Optional[str], Union[str, NamedMappingParserArgs]],
    ] = Field(default_factory=dict)
    # convertion: any -> struct
    # dict value:
    #   - None or _, keyed (dict key) value with input, {[dict key]: ...}
    #   - str, output field name, align to state field type
    #   - NamedMappingParserArgs, convert following args
    output_mapping: Dict[str, Optional[Union[str, NamedMappingParserArgs]]] = Field(
        default_factory=dict
    )

    input_expr: Optional[str] = None
    output_expr: Optional[str] = None

    def __init__(self, **data):
        super().__init__(**data)

        if isinstance(self.agent, str):
            config, _, name = self.agent.partition(":")
            if not config or not name.strip():
                raise ValueError(
                    f"Invalid agent config `{self.agent}`, should be `config_file:agent_name`"
                )
            self.agent = AgentConfig(config=config.strip(), name=name.strip())

        if get_expr_enabled():
            if self.input_mapping and self.input_expr:
                raise ValueError("Cannot specify both `input_mapping` and `input_expr`")
            if self.output_mapping and self.output_expr:
                raise ValueError(
                    "Cannot specify both `output_mapping` and `output_expr`"
                )
        else:
            if self.input_expr or self.output_expr:
                # or, just ignore it
                self.input_expr = None
                self.output_expr = None
                # raise ValueError("Cannot specify `input_expr` or `output_expr` when expression is not enabled")

        input_mapping = {}
        if isinstance(self.input_mapping, str):
            input_mapping = self.input_mapping
        elif isinstance(self.input_mapping, NamedMappingParserArgs):
            input_mapping = {None: self.input_mapping}
        else:
            for k, v in self.input_mapping.items():
                if is_primitive_field(k):
                    input_mapping[None] = v
                else:
                    input_mapping[k] = v
        self.input_mapping = rebuild_struct_mapping(input_mapping)


class BaseWorkflowModel(BaseModel):
    name: Optional[str]
    vertices: List[BaseVertex] = Field(default_factory=list[BaseVertex])
    # TODO: resolve: UserWarning: Field name "schema" shadows an attribute in parent "BaseModel
    schema_: List[FieldConfig] = Field(
        default_factory=list[FieldConfig], alias="schema"
    )
    # map schema (state) name to input name
    # convertion: any -> struct
    # same as CallableVertex.output_mapping
    input_mapping: Dict[str, Optional[Union[str, NamedMappingParserArgs]]] = Field(
        default_factory=dict
    )
    reset_state: bool = Field(default=False)

    input_expr: Optional[str] = None

    def __init__(self, **data):
        super().__init__(**data)
        if get_expr_enabled():
            if self.input_expr and self.input_mapping:
                raise ValueError("Cannot specify both `input_expr` and `input_mapping`")
        else:
            if self.input_expr:
                # or, just ignore it
                self.input_expr = None
                # raise ValueError("Cannot specify `input_expr` when expression is not enabled")


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
        if data_type
        in [
            SupportedDataType.str,
            SupportedDataType.int,
            SupportedDataType.float,
            SupportedDataType.bool,
        ]
        else DataType.struct
        if data_type == SupportedDataType.struct
        else DataType.list
        if data_type == SupportedDataType.list
        else DataType.message
        if data_type == SupportedDataType.message
        else DataType.messages
        if data_type == SupportedDataType.messages
        # for now, align with list
        else DataType.list
        if data_type == SupportedDataType.structlist
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
            if _type is str:
                _type = Annotated[str, lambda a, b: (a + "\n\n" + b).strip("\n")]
            elif _type is BaseMessage:
                # TODO: Message, to a list of Messages?
                _type = Annotated[List[BaseMessage], operator.add]
            else:
                # other types: List, Dict, Messages
                _type = Annotated[_type, operator.add]
        new_fields[field_name] = _type
    return TypedDict(model_name, new_fields)


def is_primitive_field(name: Optional[str]) -> bool:
    if name is None:
        return True
    if isinstance(name, str):
        return name == "_"
    return False


class Workflow:
    def __init__(self, model: BaseWorkflowModel):
        self.name: str = model.name or self.__class__.__name__
        self.vertices: List[BaseVertex] = model.vertices
        self.input_mapping = model.input_mapping or {}
        self.input_expr = model.input_expr
        self.schema = model.schema_ or []
        self.reset_state = model.reset_state

        self.schema_fields_map = {f.name: f for f in self.schema}
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
        fields.append(
            FieldConfig(
                **{
                    "name": NAME_BRANCH_STEPS,
                    "field_type": SupportedDataType.structlist,
                    "operator": SupportedOperators.Add,
                    "default": [],
                }
            )
        )

        # camel case the name
        model_name = f"{self.name}State"

        DynamicStateModel = typed_dict_create_model(model_name, fields=fields)
        # assign the default values and state fields
        self._default_dict = {
            f.name: f.default or _default_value_with_type(f.field_type)
            for f in self.schema
        }
        self._default_dict[NAME_BRANCH_STEPS] = []

        return DynamicStateModel

    @abstractmethod
    def build_graph(self):
        """
        Build the graph from the vertices, edges, and input/output mappings
        """

    # build graph -> compile -> build workflow
    def compile(self):
        self.build_graph()

        # NOTE: here the checkpoint is used to keep the state each turn to enter the workflow
        #   because we expect the state to be cleared after each turn
        #   If you hope to persist the state, the solution is to wrap another state saver, which sync the state from this checkpoint
        self.checkpoint = MemorySaver()
        workflow = self.graph.compile(checkpointer=self.checkpoint)
        return workflow

    def _data_to_state(
        self, data, mapping: Dict[str, Optional[Union[str, NamedMappingParserArgs]]]
    ) -> Dict[str, Any]:
        # TODO: remove the `NAME_BRANCH_STEPS` ... in some tests cases
        # if isinstance(data, dict):
        #     for k in [NAME_BRANCH_STEPS, ]:
        #         data.pop(NAME_BRANCH_STEPS, None)
        ret = parse_data_with_struct_mapping(data, mapping)
        if isinstance(data, dict):
            # add the missing fields and convert to state type
            for sname, oname in data.items():
                if sname not in mapping and sname in self.schema_fields_map:
                    field = self.schema_fields_map[sname]
                    # TODO: to be checked for any side effects.
                    # or simply: ret[sname] = data[sname]
                    ret[sname] = parse_data_with_type(
                        data[sname],
                        # align the type with default parser args
                        output_type=to_parsed_type(field.field_type),
                    )
        return ret

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

        required_keys = {self.input_mapping[f.name] for f in self.schema if f.required}

        # try expression first
        if self.input_expr:
            eval_result = eval_expr(self.input_expr, inputs)
            if isinstance(eval_result, dict):
                missing_keys = required_keys.difference(eval_result.keys())
                if len(missing_keys) > 0:
                    raise ValueError(
                        f"required fields {missing_keys} are missing in workflow {self.name}'s inputs mapping"
                    )
                # add the default values
                return {**self._default_dict, **eval_result}
            else:
                raise ValueError(
                    f"Expected expression result to be a dict, but invalid expression for workflow: {self.input_expr}"
                )

        # convert mapping to dict[str, str]
        input_mapping = {
            sn: inn.name if isinstance(inn, NamedMappingParserArgs) else inn
            for sn, inn in self.input_mapping.items()
        }
        # all required fields should be in the inputs
        try:
            missing_keys = required_keys.difference(input_mapping.values())
            if len(missing_keys) > 0:
                raise ValueError(
                    f"required fields {missing_keys} are missing in workflow {self.name}'s inputs mapping"
                )
        except KeyError:
            raise ValueError("input mapping should include all required fields")

        # re-build the output mapping first
        new_mapping: Dict[str, NamedMappingParserArgs] = {}
        for sname, oname in self.input_mapping.items():
            field = self.schema_fields_map.get(sname, None)
            if not field:
                raise ValueError(f"Unsupported state name for output mapping: {sname}")
            # must be None
            if isinstance(oname, str) or is_primitive_field(oname):
                new_mapping[sname] = NamedMappingParserArgs(
                    name=None if is_primitive_field(oname) else oname,
                    output_type=to_parsed_type(field.field_type),
                    message_role="human",
                    message_name="input",
                )
                # TODO: process the given converter function
                if field.converter:
                    pass
            elif isinstance(oname, NamedMappingParserArgs):
                oname.name = None if is_primitive_field(oname.name) else oname.name
                new_mapping[sname] = oname
            else:
                raise ValueError(f"Unsupported output mapping from `{oname}`")
        ret = self._data_to_state(inputs, new_mapping)
        # add the default values
        ret = {**self._default_dict, **ret}
        return ret

    def _enter_node_func(
        self,
        node_name: str,
        input_mapping: Dict[Optional[str], NamedMappingParserArgs],
        input_expr: Optional[str],
    ):
        def enter_node(state):
            # save a copy of the state
            state_snapshot = deepcopy(state)
            self.intermediate_steps.append(
                (
                    node_name,
                    state_snapshot,
                )
            )

            # try expression first
            if input_expr:
                eval_result = eval_expr(input_expr, state)
                return eval_result
            else:
                return parse_data_with_struct_mapping(state, input_mapping)

        return enter_node

    def _map_output_func(
        self, output_mapping: Dict, output_expr: Optional[str], node_name: str
    ):
        # map outputs to the state
        # re-build the output mapping first
        new_mapping: Dict[str, NamedMappingParserArgs] = {}
        for sname, oname in output_mapping.items():
            field = self.schema_fields_map.get(sname, None)
            if not field:
                raise ValueError(f"Unsupported state name for output mapping: {sname}")
            # must be None
            if isinstance(oname, str) or is_primitive_field(oname):
                new_mapping[sname] = NamedMappingParserArgs(
                    name=None if is_primitive_field(oname) else oname,
                    output_type=to_parsed_type(field.field_type),
                    message_role="ai",
                    message_name=node_name,
                )
                # TODO: process the given converter function
                if field.converter:
                    pass
            elif isinstance(oname, NamedMappingParserArgs):
                oname.name = None if is_primitive_field(oname.name) else oname.name
                new_mapping[sname] = oname
            else:
                raise ValueError(f"Unsupported output mapping from `{oname}`")

        def exit_node(outputs):
            # try expression first
            if output_expr:
                eval_result = eval_expr(output_expr, outputs)
                if isinstance(eval_result, dict):
                    return eval_result
                else:
                    raise ValueError(
                        f"Expected expression result to be a dict, but invalid output expression: {output_expr}"
                    )
            # convert the output to a dict
            return self._data_to_state(outputs, new_mapping)

        return exit_node

    def _to_graph_node_name(self, name):
        if name in [START, END]:
            return name
        return f"{self.name}_{name}"

    def _build_callable_node(self, v: CallableVertex):
        node_name = self._to_graph_node_name(v.name)
        if isinstance(v.agent, Callable):
            callable = v.agent
        # elif isinstance(v.agent, AgentConfig):    # not reachable
        #     callable = AgentInvoker(node_name=node_name, agent_config=v.agent).runnable
        elif isinstance(v.agent, RunnableContainer):
            callable = v.agent.build_original()
        else:
            raise ValueError(
                f"Unsupported agent type `{type(v.agent)}` in callable vertex `{v.name}`"
            )
        output_mapping = v.output_mapping or {}
        runnable = (
            RunnableLambda(
                self._enter_node_func(
                    node_name, input_mapping=v.input_mapping, input_expr=v.input_expr
                ),
                name=node_name,
            )
            | callable
            | self._map_output_func(output_mapping, v.output_expr, node_name=node_name)
        )
        return node_name, runnable

    def build_workflow(self):
        return self._enter_graph | self.compile()

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
        runnable = self._enter_graph | workflow
        return runnable.invoke(inputs, config, **kwargs)

    def stream(self, inputs, config=None, **kwargs):
        workflow = self.compile()
        # stream_mode = config.pop("stream_mode", "values")
        # workflow.stream_mode = stream_mode
        config = config or {}
        config["thread_id"] = self._gen_thread_id()
        runnable = self._enter_graph | workflow
        yield from runnable.stream(inputs, config, **kwargs)
