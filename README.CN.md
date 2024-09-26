# LcStack

LcStack 是一个 LLM 组件配置框架，用于以 LowCode 的方式快速搭建自己的 LLM Agent 和 Workflow。基于 [LangChain](https://github.com/langchain-ai/langchain) 和 [LangGraph](https://github.com/langchain-ai/langgraph) 构建，提供用户友好的配置环境和组件库来管理配置过程和 Workflow，以及提供集成其它框架的扩展能力。

## 关键特性

- 复用 LangChain 和 LangGraph 构建的组件库和开发框架
- 内置常用的 Chains 和 Agents，并支持扩展新的组件，以及适配其它框架现有组件
- 使用基础组件（LLM、PromptTemplate、Embedding、VectorStore、Memory、Tools）快速配置 Agent
- Workflow 配置支持使用已经配置好的 Agents 和其它 Workflows
- 支持重新组织输入和输出格式，适配上下游的节点的输出和输入
- 提供 Cli 库，快速运行测试

## 使用说明

### 安装

```shell
pip install -U lcstack
```

### 使用

- 基本使用：[basic.ipynb](./examples/basic.ipynb)
- 常用组件：[components.ipynb](./examples/components.ipynb)
- workflow：[workflow.ipynb](./examples/workflow.ipynb)
- workflow (hierarchy)：[workflow_hierarchy.ipynb](./examples/workflow_hierarchy.ipynb)

### 已经支持的组件

- 基础组件
    - LLM / Chat Model
    - PromptTemplate
    - Embedding
    - VectorStore
    - Memory
    - Tools
- LangChain Chains
    - LLM Chain
    - Retrival Chain
    - Document QA
    - LLM Requests Chain
    - SQL Query Chain
    - Router Chain
    - Documents Load Pipeline
- LangChain Agents
    - ReAct
    - Supervisor
    - Plan-and-Execute
- Workflow
    - Node2Node （对于非条件节点的单节点处理）
        - 节点连接
        - 条件处理分支
        - 满足条件时状态修改
        - 手动添加执行节点（不推荐）
    - TODO：Node2Nodes （非条件节点的并行处理）

