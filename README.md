# LcStack

LcStack is a configuration framework for LLM components, designed to quickly build your own LLM Agent and Workflow in a LowCode manner. Built on [LangChain](https://github.com/langchain-ai/langchain) and [LangGraph](https://github.com/langchain-ai/langgraph), it provides a user-friendly configuration environment and component library to manage the configuration process and Workflow, as well as the ability to integrate with other frameworks.

## Key Features

- Compatible with existing LangChain and LangGraph libraries and frameworks.
- Includes standard Chains and Agents, with options to add new ones or integrate with other frameworks.
- Easily set up Agents with essential components like LLM, PromptTemplate, Embedding, etc.
- Design Workflows using existing Agents and Workflows.
- Re-organize input/output to match upstream and downstream nodes.
- Offers a command-line tool for fast testing.

## Get Started 

### Installation

```shell
pip install -U lcstack
```

### Usage

- Basic usage: [basic.ipynb](./examples/basic.ipynb)
- Common components: [components.ipynb](./examples/components.ipynb)
- Workflow: [workflow.ipynb](./examples/workflow.ipynb)
- Workflow (hierarchy): [workflow_hierarchy.ipynb](./examples/workflow_hierarchy.ipynb)

### Supported Component List

- Basic components
  - LLM / Chat Model
  - PromptTemplate
  - Embedding
  - VectorStore
  - Memory / Chat History
  - Tools
- LangChain Chains
  - LLM Chain
  - Retrieval Chain
  - Document QA
  - LLM Requests Chain
  - SQL Query Chain
  - Router Chain
  - Documents Load Pipeline
  - (More coming soon)
- LangChain Agents
  - ReAct
  - Supervisor
  - Plan-and-Execute
- Workflow
  - Node2Node (for single-node processing of non-conditional nodes)
    - Node connection
    - Conditional branch processing and state modification
    - Manually adding execution nodes (not recommended)
  - TODO: Node2Nodes (parallel processing of non-conditional nodes)

### Client

  - [cli.py](./cli.py)

  
  ```shell
  # run the llm_chain in llm_chain.yaml
  python cli.py -c llm_chain.yaml -i llm_chain -q "what's your name?"
  
  # with intermediate output
  python cli.py -c llm_chain.yaml -i llm_chain -q "what's your name?" -v

  # run the wf_hierarchical_team in graph_supervisor_header.yaml, `query` as input key, and generate graph image
  python cli.py -c wf_hierarchical_team/graph_supervisor_header.yaml -i teams_manager -q "tell me three jokes first, one about dog and two about apple, but only one at a time; and then teach me 2 knowledges, one about dog, and one about apple" -I query -G teams_manager.png
  ```

  ```shell
  # for help
  python cli.py --help
  ```