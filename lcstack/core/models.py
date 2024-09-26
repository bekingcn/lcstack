import enum
from typing import Any, Dict, List, Optional, Union
from pydantic import BaseModel, Field

from lcstack.core.parsers.base import MappingParserArgs, NamedMappingParserArgs, OutputParserType

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
    output_parser_args: Optional[MappingParserArgs] = None
    
    # outputs, if dict, the key is the to key, value is the from key, following struct_mapping in the output parser
    outputs: Union[str, List[str], Dict[str, str], Dict[str, NamedMappingParserArgs]] = Field(default=[])

    # arguments to be passed to func_or_class
    kwargs: Optional[Dict[str, Any]] = {}

    def __init__(self, **data):
        super().__init__(**data)
        for k in ["output_parser_args", "outputs"]:
            data.pop(k, None)

        self.kwargs.update(data)
        
        # re-map the outputs from any type to dict
        if isinstance(self.outputs, str):
            outputs = {self.outputs: NamedMappingParserArgs(name=self.outputs, output_type=OutputParserType.pass_through)}
        if isinstance(self.outputs, list):
            outputs = {key: NamedMappingParserArgs(name=key, output_type=OutputParserType.pass_through) for key in self.outputs}
        if isinstance(self.outputs, dict):
            outputs = {key: NamedMappingParserArgs(name=val, output_type=OutputParserType.pass_through) for key, val in self.outputs.items()}
        self.outputs = outputs

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

