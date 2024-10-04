from abc import ABC, abstractmethod
import logging
from typing import List, TypedDict, Dict, Any, Optional
import uuid
from pydantic import BaseModel, Field

from lcstack.registry import get_component

from .component import Component
from .container import CHAT_HISTORY_PARAM_NAME, BaseContainer, NoneRunnableContainer, RunnableContainer
from .models import ComponentType, InitializerConfig, InitializerDataConfig, NonRunnables

class InitializerPath(BaseModel):
    path: List[str]
    
    @property
    def name(self):
        return self.path[-1]
    
    @property
    def full_name(self):
        return "::".join(self.path)
    
    @property
    def parent(self):
        if len(self.path) == 0:
            return None
        return InitializerPath(path=self.path[:-1])
    
    @classmethod
    def root(cls):
        return InitializerPath(path=[])
    
    def __str__(self):
        return "::".join(self.path)
    
    def __eq__(self, other: object) -> bool:
        if not isinstance(other, InitializerPath):
            return False
        return self.path == other.path
    
    def __hash__(self) -> int:
        return hash(self.path)
    
    def child(self, sub_path: str|list[str]):
        if isinstance(sub_path, str):
            sub_path = [sub_path]
        path = self.path + sub_path
        return InitializerPath(path=path)
    
class DpendencyInfo(BaseModel):
    name: str       # reference name
    path: InitializerPath       # dependency path
    initializer: "BaseInitializer"
    # if this is not None, it means the dependency is copied with updated kwargs
    updated_data_kwargs: Optional[Dict[str, Any]] = None

    @property
    def from_root(self) -> bool:
        return len(self.path) == 1
    
    @property
    def from_updated(self) -> bool:
        return (self.updated_data_kwargs is not None and len(self.updated_data_kwargs) > 0)
    
    @property
    def initializer_config(self) -> "InitializerConfig":
        return self.initializer.initializer_config

# TODO: make a more abstract class `BaseInitializer`, and add other subclasses like: WorkflowInitializer (under graph), etc
# something like:

class BaseInitializer(BaseModel):
    # renamed to initializer_config
    initializer_config: InitializerConfig
    path: InitializerPath
    parent: "BaseInitializer"

    children: Dict[str, "BaseInitializer"] = Field(default={})
    component: Optional[Component] = None
    # to be filled in
    # includes: children, root initializers, and copied_with_updated children
    dependencies: List[DpendencyInfo] = Field(default=[])
    # to be filled in during parse_config
    parsed_kwargs: Dict[str, Any] = Field(default={})

    def get_root(self) -> "BaseInitializer":
        # current initializer is the root
        return self.parent.get_root()

    def get_ref_initializer(self, name):
        if name in self.children:
            return self.children[name]
        elif name in self.initializer_config.children:
            child = self._init_child(name)
            return child
        else:
            return self.get_root().get_ref_initializer(name)

    def _init_child(self, name:str):
        child_component_config = self.initializer_config.children[name]
        child_initializer_name = child_component_config.initializer
        child_component = get_component(child_initializer_name)

        # TODO: get from a map really
        cls: BaseInitializer = initializer_class_map.get(child_component.component_type, ComponentInitializer)

        child = cls.from_config(child_component_config, parent=self, path=self.path.child([name]))
        child.parse_config()
        self.children[name] = child
        return child

    def parse_config(self):
        raise NotImplementedError("parse_config not implemented for base initializer")

    def build(self, name: str, **kwargs) -> BaseContainer:
        raise NotImplementedError("build not implemented for base initializer")

    @classmethod
    def from_config(cls, config: InitializerDataConfig, parent: "BaseInitializer", path: InitializerPath):
        return cls(initializer_config=config, parent=parent, path=path)


    # check value is a f-string
    # and fill in values from defaults
    def _parse_value(self, _key, _value, initializer_data, levels=None):
        if levels is None:
            levels = self.path.path
        # _value: should be one the patters:
        #   string: should be a f-string {{...}}
        #   dict: should be an initializer config {"initializer_name": "...", "initializer_data": {...}}
        #   list: should be a list of values follow above two patterns
        #   other: will be treated as original value, not processed
        # ref a initializer case 1: {{ initializer_name }}
        if isinstance(_value, str) and _value.startswith("{{") and _value.endswith("}}"):
            ref_name = _value[2:-2].strip()
            cp_name = ref_name
            args = {}
        # ref a initializer case 2: {"initializer_name": "...", "initializer_data": {...}}
        elif isinstance(_value, dict) and "initializer_name" in _value and "initializer_data" in _value:
            ref_name = _value.get("initializer_name")
            cp_name = f"{ref_name}#{uuid.uuid4().hex[0:8]}"
            args = _value.get("initializer_data", {})
            # TODO: to be checked. casscade parsing the extra args
            if args:
                for ik, iv in args.items():
                    args[ik] = self._parse_value(f"{_key}${ik}", iv, initializer_data, levels + [cp_name])
        # TODO: to be checked. casscade parsing
        elif isinstance(_value, dict):
            new_values = {}
            for vk, vv in _value.items():
                new_values[vk] = self._parse_value(f"{_key}${vk}", vv, initializer_data, levels)
            return new_values
        elif isinstance(_value, list):
            new_values = []
            for index, v in enumerate(_value):
                # TODO: check the value should be one of valid patterns                    
                new_values.append(self._parse_value(f"{_key}${index}", v, initializer_data, levels))
            return new_values
        else:
            return _value

        initializer = self.get_ref_initializer(ref_name)
        if args:
            dep_info = DpendencyInfo(
                # only marked override
                name=f"{_key}#OVERRIDE",
                path=initializer.path,
                initializer=initializer,
                updated_data_kwargs=args
            )
        else:
            dep_info = DpendencyInfo(name=_key, path=initializer.path, initializer=initializer)
        self.dependencies.append(dep_info)        
        container = initializer.build(name=_key, **args)
        return container
        
    def build(self, name: str, **kwargs) -> BaseContainer:
        """
        Args:
            name (str): TODO: the name of the in the workflow or runtime
            kwargs (dict): the extra kwargs to create lc component

        Returns:
            BaseContainer: the container of the node

        Raises:
        """

        new_kwargs = self.parsed_kwargs.copy()
        new_kwargs.update(kwargs)
        
        # prirority: config > Component > default (pass_through)
        output_mapping = self.initializer_config.data.output_mapping
        input_mapping = self.initializer_config.data.input_mapping

        # inst = self.func_or_class(**kwargs)
        if self.component.component_type in NonRunnables:
            return NoneRunnableContainer(
                name,
                self.component, 
                new_kwargs,
                shared=False)
        else:
            return RunnableContainer(
                name,
                self.component, 
                new_kwargs,
                shared=False,
                # must pop this key, cause it's common parameter as needed
                memory=new_kwargs.pop(CHAT_HISTORY_PARAM_NAME, None),
                input_mapping=input_mapping,
                output_mapping=output_mapping)

class BaseComponentInitializer(BaseInitializer):

    def parse_config(self):
        initializer_name = self.initializer_config.initializer
        component = get_component(initializer_name)
        self.component = component
        
        data = self.initializer_config.data
        parsed_data = {}
        for key, value in data.kwargs.items():
            # key as a reference to dependened initializer
            parsed_data[key] = self._parse_value(key, value, data)

        self.parsed_kwargs = parsed_data

# TODO: a map by component type
initializer_class_map = {}

class RootInitializer(BaseInitializer):
    parent: BaseInitializer | None = None
    path: InitializerPath = InitializerPath.root()

    def get_root(self):
        return self
    
    def parse_config(self):
        raise NotImplementedError("parse_config not implemented for root initializer")
    
    def get_ref_initializer(self, name):
        if name in self.children:
            initializer = self.children[name]
        elif name in self.initializer_config.children:
            initializer = self._init_child(name)
        else:
            raise ValueError(f"Unknown initializer {name}")
        return initializer
    
    # changed logic for root initializer
    def build(self, name: str, **kwargs) -> BaseContainer:
        initializer = self.get_ref_initializer(name)
        return initializer.build(name=name, **kwargs)
    
    @classmethod
    def to_root_config(cls, config: Dict[str, Any]):
        root_config = {"initializer": "__ROOT__", "data": {}, **config}
        return InitializerConfig(**root_config)

    @classmethod
    def from_config(cls, config: Dict[str, Any]):
        root_config = cls.to_root_config(config)
        return super().from_config(root_config, parent=None, path=InitializerPath.root())

class ComponentInitializer(BaseComponentInitializer):
    """
    Component initializer: all behaviors already implemented in BaseInitializer
    TODO: should move these behaviors here?
    """
    pass