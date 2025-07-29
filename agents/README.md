# Commons Agents

Agent orchestration framework with support for multiple agent types and integrations.

## Features

- **Agent Types**: ReAct, plan-and-execute, conversational, tool-using agents
- **Framework Integration**: LangChain, LlamaIndex, CrewAI, AutoGen compatibility
- **Tools & Memory**: Extensible tool system with vector and conversation memory
- **Orchestration**: Sequential, parallel, and hierarchical agent execution
- **Planning**: Goal decomposition and task planning capabilities

## Installation

```bash
# Basic installation
pip install commons-agents

# With specific frameworks
pip install commons-agents[langchain]
pip install commons-agents[llamaindex]
pip install commons-agents[crewai]
pip install commons-agents[all]
```

## Usage

### Basic Agent

```python
from commons_agents import Agent, Tool, AgentExecutor
from commons_llm import LLMProvider

# Define tools
@Tool(description="Search the web for information")
async def web_search(query: str) -> str:
    # Implementation
    return f"Search results for: {query}"

@Tool(description="Calculate mathematical expressions")
async def calculator(expression: str) -> float:
    return eval(expression)  # Simple example

# Create agent
llm = LLMProvider.create("openai", model="gpt-4")
agent = Agent(
    llm=llm,
    tools=[web_search, calculator],
    system_prompt="You are a helpful research assistant.",
)

# Execute agent
executor = AgentExecutor(agent, max_iterations=5)
result = await executor.run("What is the population of Tokyo and what's 15% of it?")
print(result)
```

### ReAct Agent

```python
from commons_agents import ReActAgent, Thought, Action, Observation

# ReAct agent with reasoning
react_agent = ReActAgent(
    llm=llm,
    tools=[web_search, calculator],
    verbose=True,  # Show reasoning steps
)

# Execute with reasoning trace
async for step in react_agent.run_iter("Solve this step by step: ..."):
    if isinstance(step, Thought):
        print(f"Thinking: {step.content}")
    elif isinstance(step, Action):
        print(f"Action: {step.tool}({step.input})")
    elif isinstance(step, Observation):
        print(f"Observation: {step.content}")

# Get final answer
result = await react_agent.run("What's the GDP per capita of Japan?")
```

### Plan-and-Execute Agent

```python
from commons_agents import PlannerAgent, ExecutorAgent, Plan

# Create planner
planner = PlannerAgent(
    llm=llm,
    planning_prompt="""
    Break down this task into clear steps:
    {task}
    
    Output a numbered list of steps.
    """
)

# Create executor
executor = ExecutorAgent(
    llm=llm,
    tools=[web_search, calculator, write_file, send_email],
)

# Plan and execute
task = "Research the top 3 AI companies and send a summary email"
plan = await planner.plan(task)

print("Plan:")
for i, step in enumerate(plan.steps, 1):
    print(f"{i}. {step}")

# Execute plan
results = []
for step in plan.steps:
    result = await executor.execute_step(step, context=results)
    results.append(result)
    print(f"✓ Completed: {step}")

print(f"Task completed successfully!")
```

### Conversational Agent

```python
from commons_agents import ConversationalAgent, ConversationMemory

# Create memory
memory = ConversationMemory(
    max_messages=50,
    summarize_after=20,  # Summarize old messages
)

# Create conversational agent
chat_agent = ConversationalAgent(
    llm=llm,
    memory=memory,
    tools=[web_search, calculator],
    personality="""
    You are a friendly AI assistant named Claude.
    You're helpful, harmless, and honest.
    """
)

# Multi-turn conversation
while True:
    user_input = input("You: ")
    if user_input.lower() == "quit":
        break
        
    response = await chat_agent.chat(user_input)
    print(f"Claude: {response}")
    
    # Memory persists across turns
    print(f"Memory size: {len(memory)}")
```

### Multi-Agent Systems

```python
from commons_agents import AgentTeam, Role

# Define roles
researcher = Agent(
    name="Researcher",
    role=Role(
        description="Research and gather information",
        goals=["Find accurate, relevant information", "Verify sources"],
        tools=[web_search, read_pdf, extract_data],
    ),
    llm=llm,
)

analyst = Agent(
    name="Analyst", 
    role=Role(
        description="Analyze data and identify insights",
        goals=["Find patterns", "Generate insights", "Create visualizations"],
        tools=[calculator, create_chart, statistical_analysis],
    ),
    llm=llm,
)

writer = Agent(
    name="Writer",
    role=Role(
        description="Create clear, engaging content",
        goals=["Write clearly", "Structure information well"],
        tools=[write_document, format_text],
    ),
    llm=llm,
)

# Create team
team = AgentTeam(
    agents=[researcher, analyst, writer],
    workflow="sequential",  # or "parallel", "hierarchical"
)

# Execute team task
task = "Create a comprehensive report on renewable energy trends"
report = await team.execute(task)
```

### Memory Systems

```python
from commons_agents import VectorMemory, EpisodicMemory, WorkingMemory

# Vector memory for semantic search
vector_memory = VectorMemory(
    embedding_model="openai/text-embedding-3-small",
    vector_store="chroma",
    collection="agent_memory",
)

# Store memories
await vector_memory.store(
    text="The capital of France is Paris",
    metadata={"type": "fact", "category": "geography"}
)

# Retrieve relevant memories
memories = await vector_memory.search(
    "What's the capital of France?",
    k=5
)

# Episodic memory for experiences
episodic_memory = EpisodicMemory(max_episodes=100)

# Record episode
episode = await episodic_memory.record_episode(
    task="Research AI companies",
    steps=[
        {"action": "web_search", "result": "Found OpenAI, Anthropic, DeepMind"},
        {"action": "analyze", "result": "All focus on AGI research"},
    ],
    outcome="success",
    learnings=["Start with company websites", "Check recent news"],
)

# Retrieve similar episodes
similar = await episodic_memory.find_similar_episodes(
    task="Research quantum computing companies"
)

# Working memory for current context
working_memory = WorkingMemory(capacity=7)  # Miller's magic number
working_memory.add("current_task", task)
working_memory.add("search_results", results)
```

### Tool Creation

```python
from commons_agents import Tool, ToolKit, ToolParameter
from typing import List, Optional

# Advanced tool definition
@Tool(
    name="database_query",
    description="Query a SQL database",
    parameters=[
        ToolParameter(
            name="query",
            type="string",
            description="SQL query to execute",
            required=True,
        ),
        ToolParameter(
            name="database",
            type="string", 
            description="Database name",
            default="default",
        ),
    ],
    examples=[
        {"query": "SELECT * FROM users LIMIT 10", "database": "app_db"},
    ],
)
async def database_query(query: str, database: str = "default") -> List[dict]:
    # Execute query
    results = await db.execute(query, database=database)
    return results

# Tool kit for related tools
web_toolkit = ToolKit(
    name="Web Tools",
    tools=[
        web_search,
        scrape_webpage,
        check_url_status,
        extract_links,
    ],
)

# Dynamic tool creation
def create_api_tool(api_name: str, base_url: str, endpoints: dict) -> Tool:
    async def api_call(endpoint: str, **params):
        url = f"{base_url}/{endpoints[endpoint]}"
        response = await http_client.get(url, params=params)
        return response.json()
    
    return Tool(
        name=f"{api_name}_api",
        description=f"Call the {api_name} API",
        func=api_call,
    )
```

### Agent Frameworks Integration

```python
# LangChain Integration
from commons_agents.integrations import LangChainAgent
from langchain.tools import DuckDuckGoSearchRun

langchain_agent = LangChainAgent(
    llm=llm,
    tools=[DuckDuckGoSearchRun()],
    agent_type="zero-shot-react-description",
)

result = await langchain_agent.run("What's the latest news on AI?")

# LlamaIndex Integration  
from commons_agents.integrations import LlamaIndexAgent
from llama_index import SimpleDirectoryReader

documents = SimpleDirectoryReader("./data").load_data()
llama_agent = LlamaIndexAgent(
    llm=llm,
    documents=documents,
    tools=[web_search, calculator],
)

answer = await llama_agent.query("Summarize the key points from the documents")

# CrewAI Integration
from commons_agents.integrations import CrewAIAdapter

crew_agent = CrewAIAdapter(
    agent=researcher,
    role="Senior Researcher",
    goal="Produce comprehensive research reports",
    backstory="You have 10 years of research experience",
)

# Use in CrewAI workflows
from crewai import Crew, Task

crew = Crew(
    agents=[crew_agent],
    tasks=[
        Task(description="Research AI safety", agent=crew_agent)
    ],
)
```

### Evaluation and Testing

```python
from commons_agents.evaluation import AgentEvaluator, EvalDataset

# Create evaluation dataset
dataset = EvalDataset([
    {
        "input": "What's 2+2?",
        "expected_output": "4",
        "expected_tools": ["calculator"],
    },
    {
        "input": "Who is the president of France?",
        "expected_output": "Emmanuel Macron",
        "expected_tools": ["web_search"],
    },
])

# Evaluate agent
evaluator = AgentEvaluator(
    metrics=["accuracy", "tool_use", "efficiency", "cost"],
)

results = await evaluator.evaluate(
    agent=agent,
    dataset=dataset,
)

print(f"Accuracy: {results['accuracy']:.2%}")
print(f"Correct tool use: {results['tool_use']:.2%}")
print(f"Average steps: {results['avg_steps']}")
print(f"Total cost: ${results['total_cost']:.4f}")

# A/B testing agents
from commons_agents.evaluation import ABTest

ab_test = ABTest(
    agent_a=react_agent,
    agent_b=plan_execute_agent,
    dataset=dataset,
)

winner = await ab_test.run(
    metrics=["accuracy", "speed", "cost"],
    confidence_level=0.95,
)

print(f"Winner: {winner.name}")
print(f"Performance difference: {winner.improvement:.2%}")
```

## Development

```bash
# Install development dependencies
pip install -e ".[dev,all]"

# Run tests
pytest

# Run specific integration tests
pytest -k langchain
pytest -k llamaindex
```