from langgraph.graph import START, END, StateGraph
from typing import Any, Callable, Dict, List, Mapping, Optional, Union
import operator

from pydantic import BaseModel, Field

from .base import NAME_BRANCH_STEPS, CallableVertex
from .base import Workflow, BaseVertex, BaseWorkflowModel
from ..core.parsers.mako import eval_expr
from ..configs import get_expr_enabled

NAME_GRAPH_DEFAULT_BRANCH = "default"
NAME_ENTER_NODE = "_enter_"

PREFIXT_STATE_KEY = "$state."
PREFIXT_STATE_EVAL_KEY = "_state_"

# Helper class to use with expr eval
class ExprMapping:
    def __init__(self, state: Dict):
        self.state = state

    def __getattr__(self, name: str) -> Any:
        return self.state[name]

class ConditionalBranch(BaseModel):
    conditions: Union[str, List[Dict[str, Any]]] = Field(default_factory=list)
    next: str = Field(default_factory=str)
    setters: Optional[List[Dict[str, Any]]] = Field(default_factory=list)

    def match_conditions(self, state) -> bool:
        if isinstance(self.conditions, str):
            # NOTE: use mako template here
            if get_expr_enabled():                
                eval_result = eval_expr(self.conditions, state)
                if eval_result is None:
                    raise ValueError(f"failed to evaluate expression or not a valid expression: {self.conditions}")
                return eval_result
        else:
            return any([c["property"] in state and state[c["property"]] == c["value"] for c in self.conditions])
    
    
    # TODO: finish this better
    @classmethod
    def parse_value(cls, state: Dict, expr: str) -> Any:
        ret = expr
        if expr.startswith(PREFIXT_STATE_KEY):
            ref = expr[len(PREFIXT_STATE_KEY):].strip()
            if ref in state:
                ret = state[ref]
            else:
                ret = None
        elif get_expr_enabled():
            # NOTE: use mako template here
            return eval_expr(expr, state)
        return ret

# TODO: support jinja template for conditions check and result setter
class JinjaBranch(BaseModel):
    conditions: str
    next: str
    setters: Optional[str] = None

class BranchsVertex(BaseVertex):
    # TODO: define ConditionalBranch and DefaultBranch instead of dict
    branchs: List[ConditionalBranch]
    # both `default` and `next` are working for default branching
    default: Optional[str] = None
    # when default matched, setters will be executed
    default_setters: Optional[List[Dict[str, Any]]] = []

    # TODO: move all conditions check and set to this class from ConditionalWorkflow

    def match(self, state) -> str:
        for b in self.branchs:
            if b.match_conditions(state):
                return b.next
        # else return 'default' as path selector
        return NAME_GRAPH_DEFAULT_BRANCH
    
    def match_and_set(self, state) -> tuple[str, Dict[str, Any]]:
        output = {}
        next_node = self.match(state)
        node_and_setters = {b.next: b.setters for b in self.branchs}
        node_and_setters[NAME_GRAPH_DEFAULT_BRANCH] = self.default_setters
        setters = node_and_setters[next_node]
        for setter in setters:
            output[setter["property"]] = ConditionalBranch.parse_value(state, setter["value"])
        return next_node, output

class ConditionalWorkflowModel(BaseWorkflowModel):
    start_vertex_name: Optional[str] = None
    vertices: List[Union[BranchsVertex, CallableVertex]] = Field(default_factory=list)

class ConditionalWorkflow(Workflow):

    def __init__(self, graph_config: ConditionalWorkflowModel):
        super().__init__(graph_config)
        self._model = graph_config

    def _branch_func(self, branch_vertex: BranchsVertex, this_name: str):
        def _func(state):
            state = state
            # check if it's from a setter node
            branch_steps = state.get(NAME_BRANCH_STEPS, None)
            if branch_steps and len(branch_steps)>0 and branch_steps[-1][0] == this_name:
                return branch_steps[-1][1]
            # from a non-setter node
            return branch_vertex.match(state)
        path_map = {}
        for branch in branch_vertex.branchs:
            path_map[branch.next] = self._to_graph_node_name(branch.next)
        # both `default` and `next` are working for default branching
        # one of `default` and `next` must be specified
        default_node = branch_vertex.default or branch_vertex.next
        if default_node:
            path_map[NAME_GRAPH_DEFAULT_BRANCH] = self._to_graph_node_name(default_node)
        else:
            # path_map[NAME_GRAPH_DEFAULT_BRANCH] = END
            raise ValueError(f"no default `branch` (or `next`) specified in vertex {this_name}")
        return _func, path_map

    def _add_conditional_edges(self, graph: StateGraph, from_node_name: str, this_name, branch_vertex: BranchsVertex):
        func, path_map = self._branch_func(branch_vertex, this_name)
        graph.add_conditional_edges(
            from_node_name,
            func,
            path_map=path_map
        )

    def _setters_func(self, this_name: str, branch_vertex: BranchsVertex):
        def _func(state):
            next_node, outputs = branch_vertex.match_and_set(state)
            outputs[NAME_BRANCH_STEPS] = [(this_name, next_node, )]
            return outputs
        return _func

    def _add_setters_and_conditional_edges(
            self, 
            graph: StateGraph, 
            from_node_name: str, 
            this_name: str, 
            branch_vertex: BranchsVertex):
        this_node_name = this_name # f"{this_name}_setters"
        setters_node = self._setters_func(this_name, branch_vertex)
        graph.add_node(this_node_name, setters_node)
        graph.add_edge(from_node_name, this_node_name)

        self._add_conditional_edges(graph, this_node_name, this_name, branch_vertex)


    def _need_setters(self, branchs: List[ConditionalBranch]):
        for branch in branchs:
            if branch.setters:
                return True
        return False

    def build_graph(self):
        """NOTE: This graph building supports only single-out graph"""

        # align to sequence graph
        # return self._build_sequence_graph()

        state_model = self.build_state_model() # .state_model
        graph = StateGraph(state_model)

        # target to list of sources,
        _edges: dict[str, list] = {}
        last_name = END
        for v in reversed(self.vertices):
            # non-conditional node, callable node
            if isinstance(v, CallableVertex):
                if not v.next:
                    v.next = last_name
                this_name = v.name
                next_name = v.next
                next_node_name = self._to_graph_node_name(next_name)
                if not next_node_name in _edges:
                    _edges[next_node_name] = []
                _edges[next_node_name].append(self._to_graph_node_name(this_name))
                last_name = this_name

        # we specify a start node, or the first node in the graph
        start_vertex_name = self._model.start_vertex_name or v.name
        start_node_name = self._to_graph_node_name(start_vertex_name)
        if not start_node_name in _edges:
            _edges[start_node_name] = []
        _edges[start_node_name].append(START)
        
        for v in self.vertices:
            if isinstance(v, BranchsVertex):
                branchs = v.branchs
                node_name = self._to_graph_node_name(v.name)
                from_node_names = _edges[node_name]
                if self._need_setters(branchs) or v.default_setters:
                    print("need setters: ", v.name)
                    for from_node_name in from_node_names:
                        self._add_setters_and_conditional_edges(graph, from_node_name, node_name, v)
                else:                    
                    for from_node_name in from_node_names:
                        self._add_conditional_edges(graph, from_node_name, None, v)
                    # not a real node, remove it after getting source node
                    _edges.pop(node_name)
            else:
                node_name, runnable = self._build_callable_node(v)
                graph.add_node(node_name, runnable)
            if v.name == start_vertex_name:
                # only one start node in this case
                self.start_nodes = [v]
        # 
        for t, ss in _edges.items():
            for s in ss:
                graph.add_edge(s, t)

        self.graph = graph
        print("build graph: ", self._model)