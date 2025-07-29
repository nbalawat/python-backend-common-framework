# Commons Workflows

Enterprise-grade workflow orchestration framework supporting complex business processes with fault tolerance, state management, and observability across multiple execution engines.

## Installation

```bash
pip install commons-workflows
```

## Features

- **Multi-Engine Support**: Temporal, Argo Workflows, AWS Step Functions, Apache Airflow
- **Workflow Patterns**: Saga, parallel execution, conditional branching, loops, fan-out/fan-in
- **Rich Activities**: HTTP calls, database operations, approval workflows, timers, signals
- **State Management**: Persistent state, automatic checkpointing, versioning, rollback
- **Fault Tolerance**: Automatic retries, compensation patterns, circuit breakers
- **Monitoring**: Real-time metrics, distributed tracing, audit logs
- **Testing**: Comprehensive testing framework with mocking and simulation
- **Human Tasks**: Approval workflows, manual interventions, escalations

## Quick Start

```python
import asyncio
from commons_workflows import Workflow, WorkflowStep, WorkflowState

# Define workflow steps
step1 = WorkflowStep("validate", "http_call", {"url": "https://api.example.com/validate"})
step2 = WorkflowStep("process", "python_function", {"function": "process_data"})
step3 = WorkflowStep("notify", "email", {"to": "admin@example.com"})

# Create workflow
workflow = Workflow("data-pipeline", [step1, step2, step3])

# Execute
result = await workflow.execute({"input": "test data"})
print(f"Workflow completed: {result.status}")
```

## Usage

### Basic Workflow

```python
from commons_workflows import Workflow, Activity, WorkflowEngine

# Define activities
@Activity
async def fetch_user(user_id: str) -> dict:
    # Fetch user from database
    return {"id": user_id, "name": "John Doe"}

@Activity
async def send_email(user: dict, subject: str, body: str) -> bool:
    # Send email
    return True

@Activity  
async def update_status(user_id: str, status: str) -> None:
    # Update user status
    pass

# Define workflow
@Workflow
async def onboarding_workflow(user_id: str) -> str:
    # Fetch user data
    user = await fetch_user(user_id)
    
    # Send welcome email
    email_sent = await send_email(
        user,
        subject="Welcome!",
        body=f"Welcome {user['name']}!"
    )
    
    if email_sent:
        # Update status
        await update_status(user_id, "onboarded")
        return "success"
    else:
        return "failed"

# Execute workflow
engine = WorkflowEngine()
result = await engine.execute(onboarding_workflow, user_id="123")
print(f"Workflow result: {result}")
```

### Saga Pattern

```python
from commons_workflows import Saga, CompensatingAction

# Define saga with compensations
saga = Saga("order-processing")

# Step 1: Reserve inventory
@saga.step(compensation="release_inventory")
async def reserve_inventory(order: dict) -> dict:
    # Reserve items
    return {"reservation_id": "res_123"}

@CompensatingAction
async def release_inventory(reservation: dict) -> None:
    # Release reserved items
    pass

# Step 2: Charge payment
@saga.step(compensation="refund_payment") 
async def charge_payment(order: dict, amount: float) -> dict:
    # Process payment
    return {"transaction_id": "txn_456"}

@CompensatingAction
async def refund_payment(transaction: dict) -> None:
    # Refund payment
    pass

# Step 3: Ship order
@saga.step
async def ship_order(order: dict, reservation: dict) -> dict:
    # Create shipment
    return {"tracking_number": "TRACK123"}

# Execute saga
async def process_order(order: dict) -> dict:
    try:
        reservation = await reserve_inventory(order)
        payment = await charge_payment(order, order["total"])
        shipment = await ship_order(order, reservation)
        return {"status": "completed", "shipment": shipment}
    except Exception as e:
        # Automatic compensation
        await saga.compensate()
        return {"status": "failed", "error": str(e)}
```

### Parallel Execution

```python
from commons_workflows import parallel, ParallelOptions

@Workflow
async def data_processing_workflow(dataset_id: str) -> dict:
    # Define parallel tasks
    tasks = [
        ("validate", validate_dataset(dataset_id)),
        ("analyze", analyze_dataset(dataset_id)),
        ("generate_report", generate_report(dataset_id)),
    ]
    
    # Execute in parallel with options
    results = await parallel(
        tasks,
        options=ParallelOptions(
            max_concurrent=3,
            fail_fast=True,  # Stop on first failure
            timeout=300,  # 5 minutes
        )
    )
    
    return {
        "validation": results["validate"],
        "analysis": results["analyze"],
        "report": results["generate_report"],
    }

# Fork/join pattern
@Workflow
async def fork_join_workflow(data: list) -> list:
    # Split data into chunks
    chunks = [data[i:i+100] for i in range(0, len(data), 100)]
    
    # Process chunks in parallel
    processed_chunks = await parallel([
        process_chunk(chunk) for chunk in chunks
    ])
    
    # Join results
    return [item for chunk in processed_chunks for item in chunk]
```

### Conditional Branching

```python
from commons_workflows import switch, case, default

@Workflow
async def approval_workflow(request: dict) -> str:
    # Get approval level based on amount
    amount = request["amount"]
    
    result = await switch(
        value=get_approval_level(amount),
        cases=[
            case("low", lambda: auto_approve(request)),
            case("medium", lambda: manager_approval(request)),
            case("high", lambda: executive_approval(request)),
        ],
        default=lambda: manual_review(request)
    )
    
    return result

# If/else pattern
@Workflow
async def conditional_workflow(user: dict) -> None:
    if user["status"] == "new":
        await send_welcome_email(user)
        await create_profile(user)
    elif user["status"] == "returning":
        await send_return_email(user)
        await update_last_login(user)
    else:
        await log_unknown_status(user)
```

### Loops and Iterations

```python
from commons_workflows import while_loop, for_each

@Workflow
async def retry_workflow(url: str) -> dict:
    # While loop with condition
    attempts = 0
    max_attempts = 5
    
    result = await while_loop(
        condition=lambda: attempts < max_attempts,
        body=async lambda: {
            attempts += 1
            response = await fetch_url(url)
            if response.status == 200:
                return response.data
            await asyncio.sleep(2 ** attempts)  # Exponential backoff
        }
    )
    
    return result

# For each pattern
@Workflow  
async def batch_workflow(user_ids: list) -> dict:
    results = await for_each(
        items=user_ids,
        activity=process_user,
        options={
            "max_concurrent": 10,
            "continue_on_error": True,
        }
    )
    
    return {
        "processed": len([r for r in results if r.success]),
        "failed": len([r for r in results if not r.success]),
    }
```

### State Management

```python
from commons_workflows import WorkflowState, checkpoint

@Workflow
class StatefulWorkflow:
    def __init__(self):
        self.state = WorkflowState()
        
    async def run(self, input_data: dict) -> dict:
        # Load previous state if resuming
        await self.state.load()
        
        # Step 1: Process data
        if not self.state.get("step1_complete"):
            result1 = await process_step1(input_data)
            self.state.set("step1_result", result1)
            self.state.set("step1_complete", True)
            await checkpoint(self.state)
        
        # Step 2: Transform data
        if not self.state.get("step2_complete"):
            result2 = await process_step2(self.state.get("step1_result"))
            self.state.set("step2_result", result2)
            self.state.set("step2_complete", True)
            await checkpoint(self.state)
        
        # Step 3: Final processing
        final_result = await process_step3(self.state.get("step2_result"))
        
        return final_result
```

### Temporal Integration

```python
from commons_workflows.temporal import TemporalWorkflow, TemporalActivity
from datetime import timedelta

@TemporalWorkflow
class OrderWorkflow:
    @TemporalActivity(
        start_to_close_timeout=timedelta(minutes=5),
        retry_policy={"maximum_attempts": 3},
    )
    async def validate_order(self, order: dict) -> bool:
        # Validate order
        return True
    
    async def run(self, order_id: str) -> str:
        # Workflow implementation
        order = await self.get_order(order_id)
        
        if await self.validate_order(order):
            payment = await self.process_payment(order)
            shipment = await self.create_shipment(order)
            
            # Wait for signal
            tracking = await self.wait_for_signal(
                "shipment_dispatched",
                timeout=timedelta(hours=24)
            )
            
            return f"Order completed: {tracking}"
        else:
            return "Order validation failed"

# Execute Temporal workflow
from temporalio.client import Client

client = await Client.connect("localhost:7233")
handle = await client.start_workflow(
    OrderWorkflow.run,
    "order_123",
    id="order-workflow-123",
    task_queue="order-processing",
)
result = await handle.result()
```

### Monitoring and Observability

```python
from commons_workflows import WorkflowMetrics, WorkflowTracer

# Enable metrics
metrics = WorkflowMetrics(
    provider="prometheus",
    endpoint="http://localhost:9090",
)

# Enable tracing
tracer = WorkflowTracer(
    provider="jaeger",
    endpoint="http://localhost:14268",
)

@Workflow(metrics=metrics, tracer=tracer)
async def monitored_workflow(data: dict) -> dict:
    # Workflow with automatic metrics and tracing
    with tracer.span("data_validation"):
        validated = await validate_data(data)
    
    with tracer.span("processing"):
        result = await process_data(validated)
    
    # Custom metrics
    metrics.increment("workflow.completed")
    metrics.gauge("processing.duration", result.duration)
    
    return result

# Query metrics
completed_count = await metrics.query(
    "workflow.completed",
    time_range="-1h",
)
```

### Testing Workflows

```python
from commons_workflows.testing import WorkflowTest, MockActivity

class TestOrderWorkflow(WorkflowTest):
    async def test_happy_path(self):
        # Mock activities
        self.mock_activity(fetch_order, return_value={"id": "123", "total": 100})
        self.mock_activity(process_payment, return_value={"status": "success"})
        
        # Execute workflow
        result = await self.execute_workflow(
            order_workflow,
            order_id="123"
        )
        
        # Assertions
        assert result["status"] == "completed"
        self.assert_activity_called(process_payment, times=1)
    
    async def test_compensation(self):
        # Mock failure
        self.mock_activity(
            ship_order,
            side_effect=Exception("Shipping failed")
        )
        
        # Execute saga
        result = await self.execute_saga(
            order_saga,
            order={"id": "123"}
        )
        
        # Verify compensations
        assert result["status"] == "compensated"
        self.assert_compensation_called(release_inventory)
        self.assert_compensation_called(refund_payment)
```

## Development

```bash
# Install development dependencies
pip install -e ".[dev,temporal,argo]"

# Run tests
pytest

# Run integration tests
pytest -m integration
```