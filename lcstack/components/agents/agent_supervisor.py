from typing import List, TypedDict, Annotated
import functools
import operator

from pydantic import BaseModel, Field
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.language_models import BaseLanguageModel

from langchain.agents import AgentExecutor, create_openai_functions_agent
from langchain.output_parsers.openai_functions import JsonOutputFunctionsParser
from langgraph.graph import END, StateGraph


def create_agent(
    llm,
    tools: list,
    system_prompt: str,
):
    """Create a function-calling agent and add it to the graph."""
    system_prompt += "\nWork autonomously according to your specialty, using the tools available to you."
    " Do not ask for clarification."
    " Your other team members (and other teams) will collaborate with you with their own specialties."
    " You are chosen for a reason! You are one of the following team members: {team_members}."

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                system_prompt,
            ),
            MessagesPlaceholder(variable_name="messages"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ]
    )
    agent = create_openai_functions_agent(llm, tools, prompt)
    executor = AgentExecutor(agent=agent, tools=tools)
    return executor


def agent_node(state, agent, name):
    result = agent.invoke(state)
    return {"messages": [AIMessage(content=result["output"], name=name)]}


def create_agent_node(name, llm, system_prompt, tools):
    agent = create_agent(llm, tools, system_prompt)
    return functools.partial(agent_node, agent=agent, name=name)


def create_team_supervisor(llm, system_prompt, members) -> str:
    """An LLM-based router."""

    if not system_prompt:
        system_prompt = (
            "You are a supervisor tasked with managing a conversation between the following workers: {team_members}. "
            + "When finished, respond with FINISH. "
            + "Given the following user request, respond with the worker to act next. "
            + "Each worker will perform a task and respond with their results and status. "
            + "Select strategically to minimize the number of steps taken."
        )

    options = ["FINISH"] + members
    function_def = {
        "name": "route",
        "description": "Select the next role.",
        "parameters": {
            "type": "object",
            "properties": {"next": {"enum": options, "type": "string"}},
            "required": ["next"],
        },
    }
    # hub: hub.pull("attercop/system-supervisor-prompt")
    # hub: hub.pull("saioru/supervisor")

    class Route(BaseModel):
        next: str = Field(description=f"Select the next role from {options}.")

    # parser = SimpleJsonOutputParser(pydantic_object=Route)
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            MessagesPlaceholder(variable_name="messages"),
            (
                "system",
                "Given the conversation above, who should act next?"
                " Or should we FINISH? Select one of options: {options}.",
                # " \nAnswer the user query.\n{format_instructions}",
            ),
        ]
    ).partial(
        options=str(options)
    )  # , team_members=", ".join(members)) #, format_instructions=parser.get_format_instructions())
    return (
        prompt
        | llm.bind_functions(functions=function_def, function_call="route")
        | JsonOutputFunctionsParser()
    )


# Research team graph state
class TeamState(TypedDict):
    # A message is added after each team member finishes
    messages: Annotated[List[BaseMessage], operator.add]
    # The team members are tracked so they are aware of
    # the others' skill-sets
    team_members: List[str]
    # Used to route work. The supervisor calls a function
    # that will update this every time it makes a decision
    next: str


def enter_chain(message: str):
    results = {
        "messages": [HumanMessage(content=message)],
    }
    return results


def join_graph(response: dict | str):
    if isinstance(response, str):
        return {"messages": [HumanMessage(content=response)]}
    return {"messages": [response["messages"][-1]]}


# member (dict): name, tools, system_prompt (if tools specified), agent (if tools not specified)
def _create_hierarchical_team(llm, supervisor, members: list, **kwargs):
    """Create a hierarchical team and add it to the graph.
    Accepts a string as input and returns a string as output.
    """
    graph = StateGraph(TeamState)
    member_names = []
    nodes = []

    for m in members:
        m_name = m.get("name", None)
        # assign a agent as the member
        if "agent" in m:
            agent_name = m["agent"]
            agent_node = kwargs.get(agent_name)
        else:
            # create an agent from tools as the member
            tools = []
            for t in m.get("tools", []):
                # TODO: not support this way in the future
                if isinstance(t, str) and t in kwargs:
                    tools.append(kwargs.get(t))
                else:
                    # TODO： check if t is a tool instance
                    tools.append(t)
            m_sys_prompt = m.get("system_prompt")
            agent_node = create_agent_node(m_name, llm, m_sys_prompt, tools)
        member_names.append(m_name)
        nodes.append(agent_node)

    enter_node = create_team_supervisor(
        llm, supervisor.get("system_prompt", None), member_names
    )
    graph.add_node("supervisor", enter_node)
    from langchain_core.runnables import RunnablePassthrough

    for m_name, agent_node in zip(member_names, nodes):
        # from create_agent_node
        if isinstance(agent_node, functools.partial):
            graph.add_node(m_name, agent_node)
        else:
            # from pre-defined agent
            graph.add_node(m_name, (agent_node | join_graph))
        graph.add_edge(m_name, "supervisor")
    cond_map = {n: n for n in member_names}
    cond_map["FINISH"] = END
    graph.add_conditional_edges(
        "supervisor",
        lambda x: x["next"],
        cond_map,
    )
    graph.set_entry_point("supervisor")
    wf = graph.compile()

    return RunnablePassthrough.assign(team_members=lambda x: member_names) | wf


# agent member: name, tools, system_prompt, llm
# supervisor: llm, system_prompt, members(names)


def create_hierarchical_team(llm, supervisor, members: list, **kwargs):
    return enter_chain | _create_hierarchical_team(llm, supervisor, members, **kwargs)


## Seperated supervisor and workers


def enter_supervisor_chain(message: str | dict):
    # print("enter_supervisor_chain message: ", message)
    if isinstance(message, dict) and "messages" in message:
        return message
    if isinstance(message, dict) and len(message) == 1:
        message = list(message.values())[0]

    results = {
        "messages": [HumanMessage(content=message)],
    }
    return results


def create_supervisor(
    llm: BaseLanguageModel,
    members,
    system_prompt=None,
    post_user_prompt=None,
    method="function_calling",
) -> str:
    """An LLM-based router."""

    if not system_prompt:
        system_prompt = (
            "You are a supervisor tasked with managing a conversation between the following workers: {team_members}. "
            + "When finished, respond with FINISH. "
            + "Given the following user request, respond with the worker to act next. "
            + "Each worker will perform a task and respond with their results and status. "
            + "Select strategically to minimize the number of steps taken."
        )

    if not post_user_prompt:
        post_user_prompt = (
            "Given the conversation above, check finished actions. Which should we FINISH?"
            + " Or which should act next? Select one of options: {options}."
            + " Make sure to return ONLY a JSON blob with key 'next'."
        )
    # " \nAnswer the user query.\n{format_instructions}",

    options = ["FINISH"] + members
    function_def = {
        "name": "route",
        "description": "Select the next role.",
        "parameters": {
            "type": "object",
            "properties": {"next": {"enum": options, "type": "string"}},
            "required": ["next"],
        },
    }
    # hub: hub.pull("attercop/system-supervisor-prompt")
    # hub: hub.pull("saioru/supervisor")

    class Route(BaseModel):
        next: str = Field(description=f"Select the next role from {options}.")

    # parser = SimpleJsonOutputParser(pydantic_object=Route)
    prompt = ChatPromptTemplate.from_messages(
        [
            # ("system", system_prompt),
            ("system", system_prompt),
            MessagesPlaceholder(variable_name="messages"),
            ("user", post_user_prompt),
        ]
    ).partial(
        options=str(options), team_members=", ".join(members)
    )  # , format_instructions=parser.get_format_instructions())
    return (
        enter_supervisor_chain
        | prompt
        | llm.with_structured_output(
            schema=function_def, method=method, include_raw=False
        )
    )
