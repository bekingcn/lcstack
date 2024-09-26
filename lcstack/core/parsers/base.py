import enum
import uuid
from logging import warning
from typing import Any, List, Dict, Optional, Set, Tuple, Union
from pydantic import BaseModel, Field

from langchain_core.messages import BaseMessage, AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.messages import ToolCall

DEFAULT_KEY_NAME = "output"
DEFAULT_MESSAGES_KEY = "messages"

DEFAULT_VALUE_ALL_MESSAGES = -99999
DEFAULT_VALUE_LAST_MESSAGE = -1

class OutputParserType(str, enum.Enum):
    """The type of the output parser.
    This enable multiple output types support for components based on lc built-in chains/agents.
    With this configuration, we should keep the built-in chains/agents' output types orginal as they are
    so that we can support multiple output types in the runtime.
    For most of cases, we should parse a `str` output or a `dict` output to a specific type.
    it's dependent on the component, this is a why we need a `supported output types` for components.
    """

    """
    Supported output types
    """
    primitive = "primitive"
    """
    primitive types like str, int, float, bool
    """
    struct = "struct"
    """
    struct outputs with key-value pairs (dict)
    """
    message = "message"
    """
    message output (BaseMessage)
    """
    messages = "messages"
    """
    messages output (List[BaseMessage])
    and slicings last n, first n, etc.
    """
    list = "list"
    """
    list output (list[str])
    """
    pass_through = "pass_through"
    """
    unknown output type
    """    
DataType = OutputParserType

class MessagesSlicingMethod(str, enum.Enum):
    all = "all"
    last = "last"
    first = "first"
    head_tail = "head_tail"
    range = "range"

class MappingParserArgs(BaseModel):
    # field name in the inputs/outputs
    # name: str     # not here
    output_type: DataType
    known_data_type: Optional[DataType] = Field(default=None)
    message_key: Optional[str] = Field(default=DEFAULT_KEY_NAME)
    messages_key: Optional[str] = Field(default=DEFAULT_MESSAGES_KEY)
    message_role: Optional[str] = Field(default="ai")
    message_name: Optional[str] = Field(default=None)
    message_index: Optional[int] = Field(default=DEFAULT_VALUE_LAST_MESSAGE)
    struct_mapping: Optional[Dict[str, Union[str, "NamedMappingParserArgs"]]] = None
    # TODO: to be supported
    messages_slicing: Optional[MessagesSlicingMethod] = None
    # head and tail of the messages for `head_tail`, and `from` and `to` for `range`
    messages_slicing_args: Optional[Tuple[int, int]] = (0, 0, )
 
# add the name mapping in/out
class NamedMappingParserArgs(MappingParserArgs):
    name: str

def default_value_with_type(_type: DataType) -> Any:
    if _type == OutputParserType.primitive:
        return ""
    elif _type == OutputParserType.struct:
        return {}
    elif _type == OutputParserType.message:
        return None
    elif _type == OutputParserType.messages:
        return []
    elif _type == OutputParserType.list:
        return []
    elif _type == OutputParserType.pass_through:
        return None
    else:
        raise ValueError(f"Unsupported output type: {_type}")

def default_output_parser_args_with_type(_type: OutputParserType) -> MappingParserArgs:
    _dict = default_output_parser_args_with_type_dict(_type)
    return MappingParserArgs(**_dict)    

def default_output_parser_args_with_type_dict(_type: OutputParserType) -> Dict:
    if _type == OutputParserType.primitive:
        return {"output_type": _type, "default_key": DEFAULT_KEY_NAME}
    elif _type == OutputParserType.struct:
        return {"output_type": _type, "struct_mapping": {}, "messages_key": DEFAULT_MESSAGES_KEY, "message_key": DEFAULT_KEY_NAME}
    elif _type == OutputParserType.message:
        return {"output_type": _type, "default_role": "ai", "default_name": "AI", "index": DEFAULT_VALUE_LAST_MESSAGE}
    elif _type == OutputParserType.messages:
        return {"output_type": _type, "default_role": "ai", "default_name": "AI", "messages_slicing": MessagesSlicingMethod.all}
    elif _type == OutputParserType.list:
        return {"output_type": _type}
    elif _type == OutputParserType.pass_through:
        return {"output_type": _type}
    else:
        raise ValueError(f"Unsupported output type: {_type}")

class BaseOutputParser:
    """Base class for an output parser."""

    def __init__(self, **kwargs: Any) -> Any:
        """Initialize the output parser."""

        # strict mode
        self.strict = kwargs.get("strict", False)

    def _message_to_str(self, message: BaseMessage) -> str:
        """Convert a message to a string."""
        return message.content

    def _messages_to_str(self, messages: List[BaseMessage], index: int = DEFAULT_VALUE_ALL_MESSAGES) -> str:
        """Convert a list of messages to a string."""
        # TODO: how to handle multiple messages, return the last one's content? or all of them?
        if index == DEFAULT_VALUE_ALL_MESSAGES:
            return ", ".join([self._message_to_str(message) for message in messages])
        else:
            return self._message_to_str(messages[index])
    
    def _list_to_str(self, messages: List[str]) -> str:
        """Convert a list of strings to a string."""
        return ", ".join(messages)
    
    def _dict_to_str(self, message: Dict[str, Any], key: str=DEFAULT_KEY_NAME) -> str:
        """Convert a dictionary to a string."""
        if key in message:
            return message[key]
        if len(message) == 1:
            warning(f"Key {key} not found in message {message}, returning the only value.")
            return list(message.values())[0]
        warning(f"Key {key} not found in message {message}, returning the keys and values.")
        return ", ".join([f"{key}: {value}" for key, value in message.items()])
    
    def _dict_to_dict(self, message: Dict, struct_mapping: Dict[str, str|Dict|"NamedMappingParserArgs"]=None) -> Dict[str, Any]:
        """Convert a dictionary to a dictionary.
        
        Args:
            struct_mapping: A dictionary that maps the keys from output keys to the keys in the input.
        """
        if not struct_mapping:
            return message
        parsed = {}
        name_mapping = {}
        for to_key, from_key in struct_mapping.items():
            # first align the mapping with dict
            if isinstance(from_key, str):
                from_key = {"name": from_key, "output_type": OutputParserType.pass_through}
            elif isinstance(from_key, MappingParserArgs):
                from_key = from_key.model_dump()
            # convert the value
            if isinstance(from_key, Dict):
                # should process nested parsing with parser args
                new_from_key = from_key.pop("name") or to_key
                if new_from_key in message:
                    parsed[to_key] = parse_data_with_type(message[new_from_key], **from_key)
                elif self.strict:
                    raise ValueError(f"Key {new_from_key} not found in message {message}.")
                else:
                    warning(f"Key {new_from_key} not matched keys in message, returning the only value or None.")
                    parsed[to_key] = parse_data_with_type(list(message.values())[0], **from_key) if len(message) == 1 else None
                name_mapping[to_key] = new_from_key
            else:
                raise ValueError(f"Unsupported struct_mapping type: {from_key}")

        if len(parsed) == 1:
            # NOTE: a trick here, if the `to_key` is None.
            #   most of the time, it get the value with the `from_key` from the `message`
            if len(struct_mapping) == 1:
                to_key, from_key = list(struct_mapping.items())[0]
                if to_key is None:
                    return parsed[to_key]

        ret = {to_key: parsed[to_key] for to_key, from_key in name_mapping.items()}
        return ret
    
    def _message_to_dict(self, message: BaseMessage, key: str=DEFAULT_KEY_NAME) -> Dict[str, Any]:
        """Convert a message to a dictionary."""
        if isinstance(message, AIMessage) and message.tool_calls:
            # NOTE: extend the tool calls
            return {key: self._message_to_str(message), "tool_calls": message.tool_calls}
        return {key: self._message_to_str(message), "tool_calls": None}
    
    def _messages_to_dict(self, messages: List[BaseMessage], key: str=DEFAULT_MESSAGES_KEY) -> Dict[str, Any]:
        """Convert a message to a dictionary."""
        return {key: messages}

    def _list_to_dict(self, messages: List[str], key: str=DEFAULT_KEY_NAME) -> Dict[str, Any]:
        """Convert a list of strings to a dictionary."""
        return {key: " ".join(messages)}
    
    def _primitive_to_dict(self, message: str, key: str=DEFAULT_KEY_NAME) -> Dict[str, Any]:
        """Convert a string to a dictionary."""
        return {key: message}
    
    def _to_message(self, message: any, type: str, name: str|None = None) -> BaseMessage:
        """Convert a primitive type to a message."""
        if type == "ai":
            return AIMessage(content=message, name=name)
        elif type == "human":
            return HumanMessage(content=message, name=name)
        elif type == "system":
            return SystemMessage(content=message, name=name)
        elif type == "tool":
            return ToolMessage(content=message, name=name)
        else:
            raise ValueError(f"Unsupported message type: {type}")
    
    def _primitive_to_message(self, message: any, type: str, name: str|None = None) -> BaseMessage:
        """Convert a string to a message."""
        return self._to_message(str(message), type, name)

    def _dict_to_message(self, message: Dict[str, Any], type: str, name: str|None = None) -> BaseMessage:
        """Convert a dictionary to a message."""
        # TODO: to be checked if this is needed
        if "tool_calls" in message and message["tool_calls"] and type == "ai":
            id_prefix = None
            for i, tool_call in enumerate(message["tool_calls"]):
                if "id" not in tool_call:
                    id_prefix = id_prefix or f"tc_{uuid.uuid4()}"
                    tool_call["id"] = f"{id_prefix}_{i}"
            return AIMessage(content="", name=name, tool_calls=message["tool_calls"])

        if len(message) == 1:
            content = list(message.values())[0]
        else:
            content = self._dict_to_str(message)
        return self._to_message(content, type, name)
    
    def _list_to_message(self, messages: List[str], type: str, name: str|None = None) -> List[BaseMessage]:
        """Convert a list of strings to a message."""
        content = self._list_to_str(messages)
        return [self._to_message(content, type, name)]
    
    def _messages_to_message(
            self, 
            messages: List[BaseMessage], 
            type: str, 
            name: str|None = None, 
            index: int = DEFAULT_VALUE_ALL_MESSAGES
        ) -> BaseMessage:
        """Convert a list of messages to a message."""
        if index == DEFAULT_VALUE_ALL_MESSAGES:
            content = self._messages_to_str(messages)
            if not type and len(messages) > 0:
                type = messages[0].type
            return self._to_message(content, type, name)
        else:
            # TODO: ignore type and name checks?
            return messages[index]
        
    def _primitive_to_messages(self, message: any, type: str, name: str|None = None) -> List[BaseMessage]:
        """Convert a string to a list of messages."""
        return [self._to_message(message, type, name)]
    
    def _dict_to_messages(self, message: Dict[str, Any], type: str, name: str|None = None) -> List[BaseMessage]:
        """Convert a dictionary to a list of messages."""
        return [self._dict_to_message(message, type, name)]
    
    def _list_to_messages(self, messages: List[str]) -> List[BaseMessage]:
        """Convert a list of strings to a list of messages."""
        raise NotImplementedError("list to messages not implemented")
    
    def _message_to_messages(self, message: BaseMessage) -> List[BaseMessage]:
        """Convert a message to a list of messages."""
        return [message]
    
    def _messages_to_messages(self, messages: List[BaseMessage], messages_slicing: MessagesSlicingMethod, messages_slicing_args: Tuple[int, int]) -> List[BaseMessage]:
        """Convert a list of messages to a list of messages."""
        if not messages_slicing or messages_slicing == MessagesSlicingMethod.all:
            return messages
        elif messages_slicing == MessagesSlicingMethod.first:
            return [messages[0]]
        elif messages_slicing == MessagesSlicingMethod.last:
            return [messages[-1]]
        elif messages_slicing == MessagesSlicingMethod.head_tail:
            if not messages_slicing_args:
                # TODO: return all messages? or raise error?
                return messages
            head, tail = messages_slicing_args
            head = max(head, 0)
            tail = max(tail, 0)
            if head + tail > len(messages):
                return messages
            else:
                return messages[:head] + messages[-tail:]
        elif messages_slicing == MessagesSlicingMethod.range:
            if not messages_slicing_args:
                # TODO: return all messages? or raise error?
                return messages
            start, end = messages_slicing_args
            # NOTE: end is inclusive
            return messages[start:end+1]
        else:
            raise ValueError(f"Unsupported messages slicing method: {messages_slicing}")

    def parse(self, output: Any) -> Any:
        """Parse the output into the desired format.

        Args:
            output: The output to parse.

        Returns:
            The parsed output.
        """
        raise NotImplementedError
    
class PrimitiveOutputParser(BaseOutputParser):
    """Parse the output as a primitive type."""

    def __init__(self, default_key: str=None) -> None:
        super().__init__()
        self.default_key = default_key

    def parse(self, output: Any) -> Any:
        """Parse the output into the desired format.

        Args:
            output: The output to parse.

        Returns:
            The parsed output.
        """
        if isinstance(output, str) or isinstance(output, int) or isinstance(output, float) or isinstance(output, bool):
            return output
        elif isinstance(output, BaseMessage):
            return self._message_to_str(output)
        elif isinstance(output, List) and len(output) > 0 and isinstance(output[0], BaseMessage):
            # default to the last message
            return self._messages_to_str(output, DEFAULT_VALUE_LAST_MESSAGE)
        elif isinstance(output, List):
            return self._list_to_str(output)
        # TODO: uncomment when dict is supported
        elif isinstance(output, dict):
            return self._dict_to_str(output, self.default_key)
        return output
    
class StructOutputParser(BaseOutputParser):
    """Parse the output as a struct type."""

    def __init__(self, struct_mapping: Dict[str, str] = None, message_key: str = DEFAULT_KEY_NAME, messages_key: str = DEFAULT_MESSAGES_KEY) -> None:
        """Initialize the StructOutputParser.
        
        Args:
            struct_mapping: The mapping from the output key to the desired input, works when the output is a dict.
            message_key: The key to use for the message, works when the output is a primitive or a list, {message_key: ...}.
            messages_key: The key to use for the messages, works when the output is a list of messages, {messages_key: list of messages}.
        """
        super().__init__()
        self.struct_mapping = struct_mapping
        self.message_key = message_key
        self.messages_key = messages_key

    def parse(self, output: Any) -> Any:
        """Parse the output into the desired format.

        Args:
            output: The output to parse.

        Returns:
            The parsed output.
        """
        if isinstance(output, BaseMessage):
            return self._message_to_dict(output, self.message_key)
        elif isinstance(output, List) and len(output) > 0 and isinstance(output[0], BaseMessage):
            # default to the last message
            return self._messages_to_dict(output, self.messages_key)
        elif isinstance(output, List):
            return self._list_to_dict(output, self.message_key)
        elif isinstance(output, dict):
            return self._dict_to_dict(output, self.struct_mapping)
        elif isinstance(output, (str, int, float, bool)):
            return self._primitive_to_dict(output, self.message_key)
        return output
    
class ListOutputParser(BaseOutputParser):
    """Parse the output as a list type."""

    def parse(self, output: Any) -> Any:
        """Parse the output into the desired format.

        Args:
            output: The output to parse.

        Returns:
            The parsed output.
        """
        if isinstance(output, List):
            return output
        elif isinstance(output, Dict):
            raise ValueError(f"Output must be a list or a string, got {type(output)}")
        return output

class MessageOutputParser(BaseOutputParser):
    """Parse the output as a message type."""

    def __init__(self, default_role: str = "ai", default_name: str = "AI", index: int = DEFAULT_VALUE_LAST_MESSAGE) -> None:
        super().__init__()
        self.default_role = default_role
        self.default_name = default_name
        self.index = index

    def parse(self, output: Any) -> Any:
        """Parse the output into the desired format.

        Args:
            output: The output to parse.

        Returns:
            The parsed output.
        """
        if isinstance(output, BaseMessage):
            return output
        elif isinstance(output, List) and len(output) > 0 and isinstance(output[0], BaseMessage):
            # default to the last message
            return self._messages_to_message(output, self.default_role, self.default_name, self.index)
        elif isinstance(output, List):
            return self._list_to_message(output, self.default_role, self.default_name)
        elif isinstance(output, Dict):
            return self._dict_to_message(output, self.default_role, self.default_name)
        elif isinstance(output, str) or isinstance(output, int) or isinstance(output, float) or isinstance(output, bool):
            return self._primitive_to_message(output, self.default_role, self.default_name)
        else:
            raise ValueError(f"Unsupported Message conversion, output type: {type(output)}")
        
class MessagesOutputParser(BaseOutputParser):
    """Parse the output as a messages type."""

    def __init__(self, 
                 default_role: str = "ai", 
                 default_name: str = "AI", 
                 messages_slicing: Optional[MessagesSlicingMethod] = None,
                 messages_slicing_args: Optional[Tuple[int, int]] = None,
                 ) -> None:
        super().__init__()
        self.default_role = default_role
        self.default_name = default_name
        self.messages_slicing = messages_slicing
        self.messages_slicing_args = messages_slicing_args

    def parse(self, output: Any) -> Any:
        """Parse the output into the desired format.

        Args:
            output: The output to parse.

        Returns:
            The parsed output.
        """
        if not output and not isinstance(output, (int, float, bool)):
            return []
        if isinstance(output, List):
            return output
        elif isinstance(output, (str, int, float, bool)):
            return self._primitive_to_messages(output, self.default_role, self.default_name)
        if isinstance(output, List):
            if len(output) == 0:
                return []
            if len(output) > 0 and isinstance(output[0], BaseMessage):
                return self._messages_to_messages(output, self.messages_slicing, self.messages_slicing_args)
            else:
                raise ValueError(f"Unsupported Messages conversion, output type: {type(output)}")
        elif isinstance(output, BaseMessage):
            return [output]
        # elif isinstance(output, List):
        #     return self._list_to_messages(output)
        elif isinstance(output, Dict):
            return self._dict_to_messages(output, self.default_role, self.default_name)
        else:
            raise ValueError(f"Unsupported Messages conversion, output type: {type(output)}")
        
def parse_data_with_type(
    data: Any, 
    output_type: DataType,
    known_data_type: Optional[DataType] = None,
    message_key: Optional[str] = DEFAULT_KEY_NAME,
    messages_key: Optional[str] = DEFAULT_MESSAGES_KEY,
    message_role: Optional[str] = "ai",
    message_name: Optional[str] = None,
    struct_mapping: Optional[Dict[str, str]] = None,
    messages_slicing: Optional[MessagesSlicingMethod] = None,
    messages_slicing_args: Optional[Tuple[int, int]] = None,
    # TODO: not supported, to remove in the future
    message_index: Optional[int] = DEFAULT_VALUE_LAST_MESSAGE,
) -> Any:
    """Parse the data into the desired format.

    Args:
        data: The data to parse.
        output_type: The output type.
        known_data_type: The known input data type.
        message_key: The message key, only used when the output type is struct, return {message_key: data}.
        messages_key: The messages key, only used when the output type is struct and data is a list of messages, return {messages_key: data}.
        message_role: The message role, only used when the output type is message.
        message_name: The message name, only used when the output type is message.
        struct_mapping: The struct mapping, only used when the output type is struct.
        messages_slicing: The messages slicing method, only used when the output type is messages.
        messages_slicing_args: The messages slicing args, only used when the output type is messages.

    Returns:
        The parsed data.
    """
    # pre-check data type if known data type is provided
    err = None
    # TODO: check the data type with known_data_type
    if False: # known_data_type:
        if known_data_type == DataType.primitive:
            if not isinstance(data, (str, int, float, bool)):
                err = f"Parse error: input must be a {known_data_type} type, got {type(data)}"
        elif known_data_type == DataType.struct:
            if not isinstance(data, Dict):
                err = f"Parse error: input must be a {known_data_type} type, got {type(data)}"
        elif known_data_type == DataType.message:
            if not isinstance(data, BaseMessage):
                err = f"Parse error: input must be a {known_data_type} type, got {type(data)}"
        elif known_data_type == DataType.messages:
            # what types are convertable to list?
            if not isinstance(data, (List, Set)) or not all(isinstance(m, BaseMessage) for m in data):
                err = f"Parse error: input must be a {known_data_type} type, got {type(data)} or not all items are message"
        elif known_data_type == DataType.list:
            if not isinstance(data, List):
                err = f"Parse error: input must be a {known_data_type} type, got {type(data)}"
        elif known_data_type == DataType.pass_through:
            pass
        if err:
            raise ValueError(err)
    
    struct_mapping = struct_mapping or {}
    if output_type == DataType.primitive:
        return PrimitiveOutputParser().parse(data)
    elif output_type == DataType.struct:
        return StructOutputParser(struct_mapping=struct_mapping, message_key=message_key, messages_key=messages_key).parse(data)
    elif output_type == DataType.message:
        # TODO: it's better to pass the name of graph node as message name
        return MessageOutputParser(default_role=message_role, default_name=message_name, index=DEFAULT_VALUE_LAST_MESSAGE).parse(data)
    elif output_type == DataType.messages:
        # convet message index to messages slicing
        if not messages_slicing:
            if message_index == DEFAULT_VALUE_ALL_MESSAGES:
                messages_slicing = MessagesSlicingMethod.all
            elif message_index == DEFAULT_VALUE_LAST_MESSAGE:
                messages_slicing = MessagesSlicingMethod.last
            else:
                messages_slicing = MessagesSlicingMethod.range
                messages_slicing_args = (message_index, message_index)
        return MessagesOutputParser(
            default_role=message_role, 
            default_name=message_name, 
            messages_slicing=messages_slicing, 
            messages_slicing_args=messages_slicing_args
        ).parse(data)
    elif output_type == DataType.list:
        return ListOutputParser().parse(data)
    elif output_type == DataType.pass_through:
        return data
    else:
        raise ValueError(f"Unsupported output parser type: {output_type}")