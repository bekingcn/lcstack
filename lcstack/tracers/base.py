from typing import Any, Dict
from langchain_core.tracers.base import BaseTracer
from langchain_core.tracers.langchain import Run
import datetime


class SerializedCallbackHandler(BaseTracer):
    """Tracer that calls a function with a single str parameter."""

    name: str = "serialized_callback_handler"
    """The name of the tracer. This is used to identify the tracer in the logs.
    Default is "serialized_callback_handler"."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        # construct a map of run ids to objects
        self.hierarchy: Dict[str, Dict] = {}

    def _persist_run(self, run: Run) -> None:
        self.save()

    def _format_datetime(self, dt: datetime.datetime) -> str:
        return dt.strftime("%Y-%m-%d %H:%M:%S.%f")

    def _on_node_create(self, parent_id: Any, node: Dict) -> None:
        parent = self.hierarchy.get(str(parent_id), None)
        if parent:
            parent["children"] = parent.get("children", []) + [node]
        self.hierarchy[str(node["run_id"])] = node

    def _on_node_outputs(
        self, node_id: Any, outputs: Any, end_time: datetime.datetime
    ) -> None:
        node = self.hierarchy.get(str(node_id), None)
        node["outputs"] = outputs
        node["end_time"] = self._format_datetime(end_time)

    def _on_node_error(self, node_id: Any, error: Any, end_time: Any) -> None:
        node = self.hierarchy.get(str(node_id), None)
        node["error"] = error
        node["end_time"] = self._format_datetime(end_time)

    # logging methods
    def _on_chain_start(self, run: Run) -> None:
        current = {
            "run_id": str(run.id),
            "run_type": run.run_type,
            "name": run.name,
            "parent": str(run.parent_run_id),
            "inputs": run.inputs,
            "start_time": self._format_datetime(run.start_time),
        }
        self._on_node_create(run.parent_run_id, current)

    def _on_chain_end(self, run: Run) -> None:
        # append outputs for this run
        self._on_node_outputs(run.id, run.outputs, run.end_time)

    def _on_chain_error(self, run: Run) -> None:
        # append outputs for this run
        self._on_node_error(run.id, run.error, run.end_time)

    def _on_llm_start(self, run: Run) -> None:
        inputs = (
            {"prompts": [p.strip() for p in run.inputs["prompts"]]}
            if "prompts" in run.inputs
            else run.inputs
        )
        current = {
            "run_id": str(run.id),
            "run_type": run.run_type,
            "name": run.name,
            "parent": str(run.parent_run_id),
            "inputs": inputs,
            "start_time": self._format_datetime(run.start_time),
        }
        self._on_node_create(run.parent_run_id, current)

    def _on_llm_end(self, run: Run) -> None:
        # append outputs for this run
        self._on_node_outputs(run.id, run.outputs, run.end_time)

    def _on_llm_error(self, run: Run) -> None:
        # append outputs for this run
        self._on_node_error(run.id, run.error, run.end_time)

    def _on_tool_start(self, run: Run) -> None:
        current = {
            "run_id": str(run.id),
            "run_type": run.run_type,
            "name": run.name,
            "parent": str(run.parent_run_id),
            "inputs": self.inputs,
            "start_time": self._format_datetime(run.start_time),
        }
        self._on_node_create(run.parent_run_id, current)

    def _on_tool_end(self, run: Run) -> None:
        # append outputs for this run
        self._on_node_outputs(run.id, run.outputs, run.end_time)

    def _on_tool_error(self, run: Run) -> None:
        # append outputs for this run
        self._on_node_error(run.id, run.error, run.end_time)

    # @abstractmethod
    def save(self) -> None:
        pass
