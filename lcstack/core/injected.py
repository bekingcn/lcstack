from typing import Any, Optional
from langchain_core.runnables import RunnableConfig
from .component import Component
from .container import BaseContainer
from .initializer import BaseInitializer
from .models import InitializerConfig, InitializerDataConfig, ComponentType


class InjectedObjComponent(Component):
    pass


class InjectedObjContainer(BaseContainer):
    def __init__(
        self, name: str, component: InjectedObjComponent
    ):
        super().__init__(name, component, {})

    def build_original(self, node_name: Optional[str] = None):
        return self.component.func_or_class()
    
    def build(self, node_name: str | None = None) -> Any:
        return self.component.func_or_class()
    
    def invoke(self, inputs: Any, config: Optional[RunnableConfig] = None):
        pass

class InjectedObjInitializer(BaseInitializer):
    def parse_config(self):
        pass
    
    # changed logic for root initializer
    def build(self, name: str, **kwargs) -> BaseContainer:
        return InjectedObjContainer(name, self.component)
    
    @classmethod
    def from_obj(cls, parent: BaseInitializer, name: str, obj: Any):
        # TODO: wrap a Callable into a Runnable?
        component = InjectedObjComponent(
            name=name, 
            component_type = ComponentType.Injected, 
            func_or_class=lambda: obj
        )
        return cls(
            initializer_config=InitializerConfig(
                initializer="injected", 
                data=InitializerDataConfig(kwargs={})
            ), 
            parent=parent, 
            path=parent.path.child([name]), 
            component=component
        )