from typing import Any, List, Literal, Tuple, Union

from langchain_core.messages import AIMessage, AnyMessage, ToolCall
from langchain_core.tools import BaseTool
from langchain_core.language_models import BaseChatModel
from langgraph.prebuilt.tool_node import ToolNode
from pydantic import BaseModel

def _enter_tool_node(data):
    """ Notes from ToolNode docstring. do something to satisfy the interface in some cases.
    Important:
        - The state MUST contain a list of messages.
        - The last message MUST be an `AIMessage`.
        - The `AIMessage` MUST have `tool_calls` populated.

    ToolCall schema:
        {
            "name": "foo",
            "args": {"a": 1},
            "id": "123" # optional
        }
    """
    print("=== _enter_tool_node", data)
    if isinstance(data, dict) and "messages" not in data and "tool_calls" in data:
        # make a AIMessage and as a last message
        tool_calls = data["tool_calls"]
        data["messages"] = [
            AIMessage(
                content="",
                tool_calls=tool_calls if isinstance(tool_calls, list) else [tool_calls],
                name = "tool_calls",
        )]
    # TODO: more conventions

    return data

class LcStackToolNode(ToolNode):
    def _parse_input(
        self,
        input: Union[
            list[AnyMessage],
            dict[str, Any],
            BaseModel,
        ],
    ) -> Tuple[List[ToolCall], Literal["list", "dict"]]:
        parsed = _enter_tool_node(input)
        return super()._parse_input(parsed)
        

def create_tool_node(tools: List[BaseTool]):
    return ToolNode(tools)

# TODO: or add a `tools` parameter for `create_llm
def create_tools_chat_model(
        chat_model: BaseChatModel, 
        tools: List[BaseTool]|ToolNode, 
        tool_choice: dict | str | Literal["auto", "any", "none"] | bool | None = None
    ):
    if isinstance(tools, ToolNode):
        tool_classes = list(tools.tools_by_name.values())
    else:
        tool_classes = tools
    return chat_model.bind_tools(tool_classes, tool_choice)