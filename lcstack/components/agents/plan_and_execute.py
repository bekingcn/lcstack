import operator
from typing import Annotated, List, Literal, Tuple, TypedDict, Union
from pydantic import BaseModel, Field

from langgraph.graph import StateGraph, START
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough


class Plan(BaseModel):
    """Plan to follow in future"""

    steps: List[str] = Field(
        description="different steps to follow, should be in sorted order"
    )


class Response(BaseModel):
    """Response to user."""

    response: str


class Act(BaseModel):
    """Action to perform."""

    action: Union[Response, Plan] = Field(
        description="Action to perform. If you want to respond to user, use Response. "
        "If you need to further use tools to get the answer, use Plan."
    )


_planner_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """For the given objective, come up with a simple step by step plan. \
This plan should involve individual tasks, that if executed correctly will yield the correct answer. Do not add any superfluous steps. \
The result of the final step should be the final answer. Make sure that each step has all the information needed - do not skip steps.""",
        ),
        ("placeholder", "{messages}"),
    ]
)

_replanner_prompt = ChatPromptTemplate.from_template(
    """For the given objective, come up with a simple step by step plan. 
This plan should involve individual tasks, that if executed correctly will yield the correct answer. Do not add any superfluous steps. 
The result of the final step should be the final answer. Make sure that each step has all the information needed - do not skip steps.

Your objective was this:
{input}

Your original plan was this:
{plan}

You have currently done the follow steps:
{past_steps}

Update your plan accordingly. If no more steps are needed and you can return to the user, then respond with that. Otherwise, fill out the plan. Only add steps to the plan that still NEED to be done. Do not return previously done steps as part of the plan."""
)


class PlanExecute(TypedDict):
    input: str
    plan: List[str]
    past_steps: Annotated[List[Tuple], operator.add]
    response: str


def create_plan_and_execute_agent(llm, tools_agent, planer_llm=None):
    planer_llm = planer_llm or llm
    planner = _planner_prompt | planer_llm.with_structured_output(Plan)
    replanner = _replanner_prompt | planer_llm.with_structured_output(Act)

    def execute_step(state: PlanExecute):
        plan = state["plan"]
        plan_str = "\n".join(f"{i+1}. {step}" for i, step in enumerate(plan))
        # TODO: if no more steps, return "No more steps to execute."
        if len(plan) == 0:
            past_steps = state["past_steps"]
            finak_answer = past_steps[-1][1]
            return {
                "past_steps": [
                    ("No more steps to execute.", "Respond final answer with Response.")
                ],
                "response": finak_answer,
            }
        task = plan[0]
        task_formatted = f"""For the following plan:
{plan_str}\n\nYou are tasked with executing step {1}, {task}.
"""
        agent_response = tools_agent.invoke({"messages": [("user", task_formatted)]})
        return {
            "past_steps": [(task, agent_response["messages"][-1].content)],
        }

    def plan_step(state: PlanExecute):
        plan = planner.invoke({"messages": [("user", state["input"])]})
        return {"plan": plan.steps}

    def replan_step(state: PlanExecute):
        output = replanner.invoke(state)
        if isinstance(output.action, Response):
            return {"response": output.action.response}
        else:
            return {"plan": output.action.steps}

    def should_end(state: PlanExecute) -> Literal["agent", "__end__"]:
        if "response" in state and state["response"]:
            return "__end__"
        else:
            return "agent"

    workflow = StateGraph(PlanExecute)
    # Add the plan node
    workflow.add_node("planner", plan_step)
    # Add the execution step
    workflow.add_node("agent", execute_step)
    # Add a replan node
    workflow.add_node("replan", replan_step)
    workflow.add_edge(START, "planner")
    # From plan we go to agent
    workflow.add_edge("planner", "agent")
    # From agent, we replan
    workflow.add_edge("agent", "replan")
    workflow.add_conditional_edges(
        "replan",
        # Next, we pass in the function that will determine which node is called next.
        should_end,
    )

    # meaning you can use it as you would any other runnable
    compiled_workflow = workflow.compile()

    return compiled_workflow


# This is a simple agent without replanning
def create_react_with_plan_agent(llm, tools_agent, planer_llm=None):
    planer_llm = planer_llm or llm
    planner = _planner_prompt | planer_llm.with_structured_output(Plan)

    def execute_step(state):
        question = state["input"]
        plan = state["plan"]
        plan_str = "\n".join(f"{i+1}. {step}" for i, step in enumerate(plan.steps))
        task_formatted = f"""Original objective: {question}

Planned steps:
{plan_str}

You are tasked with these steps one at a time.
"""
        agent_response = tools_agent.invoke({"messages": [("user", task_formatted)]})
        return {
            "response": agent_response["messages"][-1].content,
        }

    chain = (
        RunnablePassthrough.assign(
            plan=(lambda x: {"messages": [("user", x["input"])]}) | planner
        )
        | execute_step
    ).with_config(run_name="react_with_plan")
    return chain


# Here we have a plan and execute from langchain implementation
def create_lc_plan_and_execute_agent(llm, tools, planer_llm=None, **kwargs):
    from langchain_experimental.plan_and_execute import (
        load_agent_executor,
        load_chat_planner,
        PlanAndExecute,
    )

    planer_llm = planer_llm or llm

    planner = load_chat_planner(planer_llm)

    _tools = []
    for t in tools:
        # TODO: not support this way in the future
        if isinstance(t, str) and t in kwargs:
            _tools.append(kwargs.get(t))
        else:
            # TODO： check if t is a tool instance
            _tools.append(t)
    executor = load_agent_executor(llm, _tools)

    output_key = "response" if "output_key" not in kwargs else kwargs.pop("output_key")
    plan_and_execute = PlanAndExecute(
        planner=planner, executor=executor, output_key=output_key, **kwargs
    )

    return plan_and_execute
