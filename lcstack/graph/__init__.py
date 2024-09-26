import enum
from typing import Dict

from .conditional import ConditionalWorkflow, Workflow, ConditionalWorkflowModel
from langchain_core.runnables import Runnable

class WorkflowType(str, enum.Enum):
    Conditional = "conditional"

# def create_workflow(workflow_type: WorkflowType=WorkflowType.Conditional, inline: Dict=None, file: str=None) -> Workflow:
#     from lcstack import get_config_root
#     if inline and isinstance(inline, dict):
#         wf_config = inline
#     elif file and isinstance(file, str):
#         import yaml
#         wf_config = yaml.safe_load(open(get_config_root() / file, "r"))
#     else:
#         raise ValueError("Either inline (dict) or file (string) must be provided for workflow creation.")
#     wf_config = ConditionalWorkflowModel(**wf_config)
#     if workflow_type == WorkflowType.Conditional:
#         wf = ConditionalWorkflow(wf_config)
#     else:
#         raise ValueError(f"Unsupported workflow type: {workflow_type}")
#     return wf
# 
# def create_workflow_agent_deprecated(workflow_type: WorkflowType=WorkflowType.Conditional, inline: Dict=None, file: str=None) -> Runnable:
#     workflow = create_workflow(workflow_type, inline, file)
#     return workflow._enter_graph | workflow.compile()

def create_workflow_agent(workflow_type: WorkflowType=WorkflowType.Conditional, inline: Dict=None, file: str=None) -> Runnable:
    workflow_type = workflow_type or WorkflowType.Conditional
    if workflow_type == WorkflowType.Conditional:
        wf_config = ConditionalWorkflowModel(**inline)
        workflow = ConditionalWorkflow(wf_config)
    else:
        raise ValueError(f"Unsupported workflow type: {workflow_type}")
    return workflow._enter_graph | workflow.compile()

__all__ = [
    "WorkflowType", 
    "create_workflow_agent"
]
