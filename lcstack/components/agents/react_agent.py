from langgraph.prebuilt import ToolNode, create_react_agent
from langgraph.graph import END, MessageGraph
from langchain_core.messages import HumanMessage


# TODO: how and which to use them
# - react_agent: langchain agent with StateGraph
# - tools_agetn: custom agent with MessageGraph
# - create_tools_agent_node (agent_supervisor.py): run agent with tools with ToolExecutor
def react_agent(
    llm,
    messages_modifier: str = None,
    checkpointer=None,
    debug: bool = False,
    wrapping=True,
    **kwargs,
):
    tools = []
    if "tools" in kwargs:
        tools = kwargs["tools"]
    else:
        # TODO: not support this way in the future
        for k, v in kwargs.items():
            if k.startswith("tool_"):
                tools.append(v)
    agent = create_react_agent(
        llm,
        tools,
        messages_modifier=messages_modifier,
        checkpointer=checkpointer,
        debug=debug,
    )
    if wrapping:
        return (
            (  # enter conversation
                lambda x: {"messages": [HumanMessage(content=x)]}
                if isinstance(x, str)
                else x
            )
            | agent
            | (  # exit conversation
                lambda x: x.get("messages", [])[-1]
            )
        )
    return agent


def should_continue(messages):
    last_message = messages[-1]
    # If there is no function call, then we finish
    if not last_message.tool_calls:
        return END
    else:
        return "tools"


def tools_agent(llm, **kwargs):
    # Define a new graph
    workflow = MessageGraph()
    # tools = [TavilySearchResults(max_results=1)]
    tools = []
    for k, v in kwargs.items():
        if k.startswith("tool_"):
            tools.append(v)
    llm_with_tools = llm.bind_tools(tools)
    workflow.add_node("agent__llm", llm_with_tools)
    workflow.add_node("agent__tools", ToolNode(tools))

    workflow.set_entry_point("agent__llm")

    # Conditional agent -> action OR agent -> END
    workflow.add_conditional_edges(
        "agent__llm", should_continue, {"tools": "agent__tools", END: END}
    )

    # Always transition `action` -> `agent`
    workflow.add_edge("agent__tools", "agent__llm")

    # Setting the interrupt means that any time an action is called, the machine will stop
    app = workflow.compile()
    return app
