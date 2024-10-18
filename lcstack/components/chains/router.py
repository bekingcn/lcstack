from typing import Any, Dict, List, Optional

from langchain_core.runnables import Runnable, RunnablePassthrough
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.language_models import BaseLanguageModel


def create_router_chain(
    llm: BaseLanguageModel,
    method: str = "function_calling",
    system_prompt: Optional[str] = None,
    default_members: List[str] = [],
) -> Runnable[Dict[str, Any], Dict[str, Any]]:
    route_system = """Answer the following question.
Route the user's query to either the {members_str} worker. 
Make sure to return ONLY a JSON blob with keys 'destination'.\n\n"""

    route_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt or route_system),
            ("human", "{query}"),
        ]
    )

    def router_query_schema(x):
        return {
            "name": "RouteQuery",
            "description": "Route query to destination.",
            "parameters": {
                "type": "object",
                "properties": {"destination": {"enum": x["members"], "type": "string"}},
                "required": ["destination"],
            },
        }

    route_chain = RunnablePassthrough.assign(
        members=lambda x: x.get("members", default_members)
    ).assign(
        members_str=lambda x: ", ".join(x["members"][:-1]) + " or " + x["members"][-1]
    ) | (
        lambda x: route_prompt
        | llm.with_structured_output(
            schema=router_query_schema(x), method=method, include_raw=False
        )
    )
    return route_chain
