from typing import Callable

from lcstack.core.container import BaseContainer
from ..registry import get_component
from ..core.initializer import BaseInitializer, DpendencyInfo, InitializerPath
from .base import AgentConfig, CallableVertex

# TODO： global cache, any side effects will be lost?
cached_configs = {}


class WorkflowInitializer(BaseInitializer):
    def parse_config(self):
        initializer_name = self.initializer_config.initializer
        component = get_component(initializer_name)
        self.component = component

        initializer_data = self.initializer_config.data
        # TODO: process inline and file parsing
        # after the workflow is created, the dependencies will be resolved to BaseContainer
        from lcstack import get_config_root

        _key = None
        if "inline" in initializer_data.kwargs:
            wf_config = initializer_data.kwargs.get("inline")
            _key = "inline" + "#" + wf_config.get("name", "Unknown")[0:4]
        elif "file" in initializer_data.kwargs:
            # TODO: here simple load a yaml file. need to support !SET, !INC, !ENV for a workflow config?
            import yaml

            file = initializer_data.kwargs.get("file")
            wf_config = yaml.safe_load(open(get_config_root() / file, "r"))
            _key = "file"
        else:
            raise ValueError(
                "Either inline (dict) or file (string) must be provided for workflow creation."
            )
        # process {{ ref }} patterns
        parsed_workflow = self._parse_value(_key, wf_config, initializer_data)

        workflow_type = initializer_data.kwargs.get("workflow_type", None)

        # process `agent config` patterns in `vertices`, which is common for all workflow models
        # must include `vertices` field, process the nested `agent` field
        for i, v in enumerate(parsed_workflow["vertices"]):  # parsed_model.vertices:
            # TODO: have to use model_validate to get the Vertex?
            try:
                cv = CallableVertex.model_validate(v)
            except Exception:
                cv = None

            if cv:
                if isinstance(cv.agent, AgentConfig):
                    agent_config = cv.agent
                    # TODO: load from config, better way to do this
                    # node_name, use the vertex name as node name? without to graph node name?
                    # TODO: process agent.kwargs
                    agent_initializer = self._load_from_config_file(cv.agent)
                    agent = agent_initializer.build(name=cv.name, **cv.agent.kwargs)
                    v["agent"] = agent
                    # add new dependency
                    dep_info = DpendencyInfo(
                        name=f"{_key}$vertices${i}$agent",
                        path=InitializerPath(
                            path=[f"{agent_config.config}:{agent_config.name}"]
                        ),
                        initializer=agent_initializer,
                    )
                    # TODO: add a node in children? which represent the agent from external file?
                    self.dependencies.append(dep_info)
                elif isinstance(cv.agent, Callable):
                    # Callable
                    pass
                elif isinstance(cv.agent, BaseContainer):
                    # BaseContainer
                    pass
                else:
                    raise ValueError(
                        f"Unsupported agent type after base parsing: {type(cv.agent)}"
                    )

        # this will be used in `create_workflow`
        self.parsed_kwargs = {"workflow_type": workflow_type, "inline": parsed_workflow}
        return self.parsed_kwargs

    def _load_from_config_file(self, agent_config: AgentConfig):
        """Load the agent from the config file

        TODO: should avoid dead loops which a agent refer to this workflow
        """
        from ..base import LcStack
        from ..configs import get_config_root

        agent_name: str = agent_config.name
        agent_config: str = agent_config.config
        if False:  # agent_config in cached_configs:
            lcs = cached_configs[agent_config]
        else:
            # lcs = LcStackBuilder.from_yaml(get_config_root() / agent_config).with_env().with_settings().build()
            lcs = LcStack.from_yaml(
                get_config_root() / agent_config, env=True, secret=True, settings=True
            )
            cached_configs[agent_config] = lcs

        initializer = lcs.get_initializer(agent_name)
        return initializer
