# Python Commons Examples

This directory contains example applications demonstrating the use of Python Commons modules.

## Examples

### 1. Research Assistant
A research assistant that uses LLM and agent modules to search, analyze, and summarize information.

```bash
cd research-assistant
pip install -r requirements.txt
python main.py "Research the latest developments in quantum computing"
```

### 2. Customer Service Bot
An intelligent customer service bot using conversational agents and workflow management.

```bash
cd customer-service
pip install -r requirements.txt
python main.py
```

### 3. Data Pipeline
ETL pipeline example using the pipelines module with multiple data sources.

```bash
cd data-pipeline
pip install -r requirements.txt
python main.py --source s3://bucket/data --output s3://bucket/processed
```

### 4. Agent Workflow
Complex multi-agent workflow for document processing and analysis.

```bash
cd agent-workflow
pip install -r requirements.txt
python main.py process --input documents/ --output results/
```

## Running Examples

Each example includes:
- `README.md` - Detailed documentation
- `requirements.txt` - Dependencies
- `config.yaml` - Configuration
- `main.py` - Main application
- `tests/` - Unit tests

## Creating Your Own Examples

1. Copy the `example-template/` directory
2. Update `requirements.txt` with needed commons modules
3. Implement your application logic
4. Add tests and documentation

## Contributing Examples

We welcome example contributions! Please:
1. Follow the existing structure
2. Include comprehensive documentation
3. Add unit tests
4. Ensure examples are self-contained