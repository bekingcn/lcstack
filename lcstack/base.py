from copy import deepcopy
import logging
from typing import Callable

from langchain_core.runnables import Runnable

from lcstack.core.models import InitializerConfig
from lcstack.lcstack_builder import YamlBuilder
from .registry import get_component
from .core.component import Component
from .core.container import BaseContainer

logging.basicConfig(format="%(asctime)s - %(name)s - %(levelname)s - %(message)s") # , level=logging.DEBUG)
logger = logging.getLogger("LcStack")

NAME_INITIALIZER = "initializer"
# TODO: change to "inputs"?
NAME_ARGUMENT_DATA = "data"

from .core.initializer import BaseInitializer, RootInitializer
class LcStack(RootInitializer):
    """Responsible for building the LcStack config to initializers graph"""
    @property
    def initializers(self):
        return self.children
    
    def get_initializer(self, name):
        return self.get_ref_initializer(name)
    
    def get(self, name: str, **kwargs) -> Runnable|Callable:
        """Get the cotainer by name, which is used to create the runnable or callable
        
        Args:
            name (str): the name of the initializer
            kwargs (dict): the extra kwargs to create container, which will be updated during the build process

        Returns:
            BaseContainer: the container of the node
            """
        initializer = self.get_ref_initializer(name)
        return initializer.build(name=name, **kwargs).build()
    
    def prebuild(self):
        for name in self.initializer_config.children:
            if name not in self.initializers:
                self.get_ref_initializer(name)
        return self
        
    def _get_nodes(self, initializer: BaseInitializer, nodes: dict):
        for cn, ci in initializer.children.items():
            full_name = ci.path.full_name
            nodes[full_name] = f"{full_name}\n{_node_content(ci.initializer_config)}"
            self._get_nodes(ci, nodes)

    def _get_edges(self, initializer: BaseInitializer, edges: list):
        for di in initializer.dependencies:
            edges.append((di.path.full_name, initializer.path.full_name, di.name))
        for cn, ci in initializer.children.items():
            self._get_edges(ci, edges)

    # draw the graph with mermaid
    def draw_graph(self) -> str:
        """draw a mermaid graph of the LcStack components
        
        Returns:
            str: the mermaid graph
        """
        from .utils import draw_mermaid
        nodes = {}
        edges = []
        self._get_nodes(self, nodes)
        for cn, ci in self.children.items():
            self._get_edges(ci, edges)
        
        # import json
        # json.dump(nodes, open("nodes.json", "w"))
        # json.dump(edges, open("edges.json", "w"))
        return draw_mermaid(nodes, edges)
    
    def draw_graph_png(self) -> bytes:
        """draw a mermaid graph of the LcStack components in png format

        Returns:
            bytes: the mermaid graph in png format
        """
        from .utils import  draw_mermaid_png
        return draw_mermaid_png(self.draw_graph())
    
    @classmethod
    def from_yaml(cls, 
                  yaml: str, 
                  env: bool=True, 
                  secret: bool=True, 
                  settings: str|bool=True, 
                  other_config:dict|None=None
    ) -> "LcStack":
        """build the LcStack from yaml config file
        
        Args:
            yaml (str): the yaml config file
            env (bool, optional): whether to load env. Defaults to True.
            secret (bool, optional): whether to load secret. Defaults to True.
            settings_file (str|None, optional): the settings file to load. Defaults to None.
            other_config (dict|None, optional): other config to merge with parsed config. Defaults to None.

        Returns:
            LcStack: the LcStack instance
        """
        builder = (
                YamlBuilder
                   .from_yaml(yaml)
                   .with_env(env=env)
                   .with_secret(secret=secret)
                   .with_settings(settings)
                   .merge_with(other_config)
        )
        config = builder._build_config()
        if config is None:
            raise ValueError("Invalid config or config file")
        return cls.from_config(config)
    
class LcStackBuilder(YamlBuilder):
    def build(self) -> LcStack:
        self.config = self._build_config()
        if self.config is None:
            raise ValueError("Invalid config or config file")
        return LcStack.from_config(self.config)

def _node_content(node: InitializerConfig):
    content = f"initializer: {node.initializer}"
    comp = get_component(node.initializer)
    if node.data.kwargs and "provider" in node.data.kwargs:
        content += f"\nprovider: {node.data.kwargs['provider']}_{node.data.kwargs.get('tag', 'none')}"
    elif isinstance(comp, Component):
        cls_name = comp.func_or_class.__name__
        content += f"\nname: {cls_name}"
    else:
        raise ValueError(f"Unknown initializer: {node['initializer']}")
    return content