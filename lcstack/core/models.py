import enum
from typing import Any, Dict, List, Optional, Union
from pydantic import BaseModel, Field

from lcstack.core.parsers.base import MappingParserArgs, NamedMappingParserArgs, OutputParserType, rebuild_struct_mapping

class ComponentType(str, enum.Enum):
    Unknown = "Unknown"
    # Langchain supported components
    Chain = "Chain" # Chain or Runnable Sequence
    LLM = "LLM" # TODO: remove
    ChatModel = "ChatModel" # TODO: remove
    LanguageModel = "LanguageModel" # LLM or ChatModel
    Retriever = "Retriever"
    PromptTemplate = "PromptTemplate"
    Tool = "Tool"
    ToolNode = "ToolNode"
    # none runnables
    Embeddings = "Embeddings"
    VectorStore = "VectorStore"
    TextSplitter = "TextSplitter"
    DocumentTransformer = "DocumentTransformer" # TODO: remove
    DocumentLoader = "DocumentLoader"
    ChatMessageHistory = "ChatMessageHistory"
    # TODO: DocumentCompressor, OutputParser (runnable)
    # helper components
    DataConverter ="DataConverter"

    # Langchain Graph components
    StateGraph = "StateGraph"
    MessageGraph = "MessageGraph"

LanguageModels = [
    ComponentType.LanguageModel,
    ComponentType.LLM, 
    ComponentType.ChatModel
]

ChainableRunnables = [
            ComponentType.Chain, 
            ComponentType.StateGraph, 
            ComponentType.MessageGraph, 
            ComponentType.Retriever,
            ComponentType.PromptTemplate,
            ComponentType.DataConverter,
            ComponentType.Tool,
            ComponentType.ToolNode
]

NonRunnables = [
    ComponentType.Embeddings,
    ComponentType.VectorStore,
    ComponentType.TextSplitter,
    ComponentType.DocumentTransformer,
    ComponentType.DocumentLoader,
    ComponentType.ChatMessageHistory
]

class ComponentParameter(BaseModel):
    # _type: ComponentType = ComponentType.Unknown
    field_type: str = "str"
    component_type: Optional[ComponentType] = ComponentType.Unknown

class InitializerDataConfig(BaseModel):
    # output_mapping on config level
    # key is schema field name, value is input or output name
    # convertion: any -> any
    # str, pop from input as node input
    # dict key:
    #   - None or _, as above, pop from input as node input
    # dict value:
    #   - None, keyed (with dict key) input value
    #   - NamedMappingParserArgs, convert following args. the name could be None or _ following above rules
    output_mapping: Union[str, MappingParserArgs, Dict[Optional[str], Optional[Union[str, NamedMappingParserArgs]]]] = Field(default_factory=dict)
    # support input_mapping on config level
    # following above rules: any -> any
    input_mapping: Union[str, Dict[Optional[str], Optional[Union[str, NamedMappingParserArgs]]]] = Field(default_factory=dict, )

    # arguments to be passed to func_or_class
    kwargs: Optional[Dict[str, Any]] = {}

    def __init__(self, **data):
        super().__init__(**data)
        for k in ["input_mapping", "output_mapping"]:
            data.pop(k, None)

        self.kwargs.update(data)
        
        # rebuild output_mapping
        self.output_mapping = rebuild_struct_mapping(self.output_mapping, check_args=True)
            
        # rebuild input_mapping, without check_args (MappingParserArgs)
        self.input_mapping = rebuild_struct_mapping(self.input_mapping, check_args=True)

# TODO: Component config model
class InitializerConfig(BaseModel):
    initializer: str
    data: InitializerDataConfig = Field(default_factory=InitializerDataConfig)

    children: Optional[Dict[str, "InitializerConfig"]] = None

    def __init__(self, **data):
        super().__init__(**data)

        for k in ["initializer", "data"]:
            data.pop(k, None)

        # in this case, we allow to specify children in the current level and `children` level
        children_configs = data.pop("children", {})
        children_configs.update(data)
        self.children = {k: InitializerConfig(**v) for k, v in children_configs.items()}

