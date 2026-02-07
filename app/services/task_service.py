import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from app.exceptions.base import AppException
from app.exceptions.validation import ValidationError
from app.models.task import Task
from app.models.trace import ExecutionTrace
from app.observability.metrics import MetricsCollector
from app.observability.tracer import Tracer
from app.repositories.task_repository import TaskRepository
from app.repositories.trace_repository import TraceRepository
from app.schemas.task_schema import TaskCreate
from app.services.decision_engine import DecisionEngine
from app.services.llm_client import LLMClient
from app.utils.logger import get_logger

logger = get_logger(__name__)


class TaskResult:
    def __init__(self, task: Task, success: bool, traces: List[Dict[str, Any]] = None):
        self.task = task
        self.success = success
        self.traces = traces or []


class TaskService:
    def __init__(self):
        self.task_repo = TaskRepository()
        self.trace_repo = TraceRepository()
        self.decision_engine = DecisionEngine()
        self.llm_client = LLMClient()
        self.tracer = Tracer()
        self.metrics = MetricsCollector()

    def execute_task(self, task_data: TaskCreate) -> TaskResult:
        """Execute a task with full observability."""
        start_time = time.time()
        task = None
        
        try:
            # Validate task data
            self._validate_task_data(task_data)
            
            # Create and save task
            task = self._create_task_record(task_data)
            
            # Start execution with tracing
            with self.tracer.start_trace(
                "task_execution", 
                {
                    "task_id": task.id, 
                    "task_type": task.task_type,
                    "input_size": len(str(task.input_data)) if task.input_data else 0,
                }
            ):
                # Store task_id in trace context for association
                self.tracer.context.current["task_id"] = task.id
                
                # Mark task as running
                task.start_execution()
                self.task_repo.save(task)
                logger.info("Task execution started", extra={"task_id": task.id})
                
                # Execute task logic
                result = self._execute_task_logic(task, task_data)
                
                # Complete task
                task.complete_execution(
                    success=result["success"],
                    output=result.get("output"),
                    error=result.get("error")
                )
                self.task_repo.save(task)
                
                # Record metrics
                self.metrics.record_task_completion(
                    task.task_type, 
                    task.status, 
                    task.execution_time_ms or 0
                )
                
                logger.info(
                    "Task execution completed",
                    extra={
                        "task_id": task.id,
                        "status": task.status,
                        "execution_time_ms": task.execution_time_ms,
                        "success": result["success"],
                    },
                )
                
                return TaskResult(
                    task=task,
                    success=result["success"],
                    traces=self.tracer.get_current_traces(),
                )
                
        except (AppException, ValidationError) as e:
            # Our custom exceptions - they already log themselves
            logger.error(
                f"Task execution failed with {e.__class__.__name__}: {e.message}",
                extra={"task_type": task_data.task_type, **e.extra},
            )
            
            # Update existing task if it exists, otherwise create new
            if task:
                # UPDATE EXISTING TASK instead of creating new one
                task = self._update_task_to_failed(task, start_time, e)
            else:
                # Only create new task if one doesn't exist
                task = self._handle_task_failure(task_data, start_time, e)
                
            return TaskResult(task=task, success=False, traces=self.tracer.get_current_traces())
            
        except Exception as e:
            # Unexpected exceptions
            logger.error(
                f"Unexpected error in task execution: {str(e)}",
                extra={
                    "task_type": task_data.task_type,
                    "original_exception_type": type(e).__name__,
                },
                exc_info=True,
            )
            
            # Wrap in AppException for consistent handling
            wrapped_exception = AppException(
                message=f"Unexpected error during task execution: {str(e)}",
                extra={
                    "task_type": task_data.task_type,
                    "original_exception_type": type(e).__name__,
                    "original_exception": str(e),
                    "unexpected_error": True,
                }
            )
            
            # Update existing task if it exists, otherwise create new
            if task:
                task = self._update_task_to_failed(task, start_time, wrapped_exception)
            else:
                task = self._handle_task_failure(task_data, start_time, wrapped_exception)
                
            return TaskResult(task=task, success=False, traces=self.tracer.get_current_traces())

    def _update_task_to_failed(self, task: Task, start_time: float, exception: Exception) -> Task:
        """Update an existing task to failed status."""
        try:
            execution_time = int((time.time() - start_time) * 1000)
            
            # Mark as failed
            task.status = "failed"
            task.error_message = str(exception)
            task.completed_at = datetime.now(timezone.utc)
            
            # Calculate execution time
            if task.started_at:
                if task.started_at.tzinfo is None:
                    task.started_at = task.started_at.replace(tzinfo=timezone.utc)
                if task.completed_at.tzinfo is None:
                    task.completed_at = task.completed_at.replace(tzinfo=timezone.utc)
                
                task.execution_time_ms = int(
                    (task.completed_at - task.started_at).total_seconds() * 1000
                )
            else:
                task.execution_time_ms = execution_time
            
            # Save the updated task
            self.task_repo.save(task)
            
            logger.info("Task updated to failed", extra={
                "task_id": task.id,
                "execution_time_ms": task.execution_time_ms,
            })
            
            return task
            
        except Exception as e:
            logger.error(
                f"Failed to update task to failed state: {str(e)}",
                extra={
                    "task_id": task.id if task else None,
                    "update_error": True,
                },
                exc_info=True,
            )
            return task

    def _validate_task_data(self, task_data: TaskCreate) -> None:
        """Validate task creation data."""
        if not task_data.task_type:
            raise ValidationError(
                message="Task type is required",
                field="task_type",
                value=task_data.task_type,
                validation_rules={"required": True, "type": "string"},
                log=False,
            )
        
        # Additional business logic validation
        if task_data.input_data and len(str(task_data.input_data)) > 10000:
            raise ValidationError(
                message="Input data too large (max 10000 characters)",
                field="input_data",
                value=f"{len(str(task_data.input_data))} characters",
                validation_rules={"max_length": 10000},
                log=False,
            )
    
    def _create_task_record(self, task_data: TaskCreate) -> Task:
        """Create and save a task record."""
        try:
            task = Task(
                task_type=task_data.task_type,
                input_data=task_data.input_data,
                parameters=task_data.parameters or {},
                status="pending",
            )
            
            self.task_repo.save(task)
            logger.info("Task created", extra={
                "task_id": task.id, 
                "task_type": task.task_type,
                "parameters_count": len(task.parameters),
            })
            
            return task
            
        except Exception as e:
            raise AppException(
                message=f"Failed to create task record: {str(e)}",
                extra={
                    "task_type": task_data.task_type,
                    "original_exception_type": type(e).__name__,
                    "database_error": True,
                }
            ) from e
    
    def _handle_task_failure(
        self, 
        task_data: TaskCreate, 
        start_time: float, 
        exception: Exception
    ) -> Task:
        """Handle task failure by creating or updating task record."""
        try:
            execution_time = int((time.time() - start_time) * 1000)
            
            # Try to create a failed task record
            task = Task(
                task_type=task_data.task_type,
                input_data=task_data.input_data,
                parameters=task_data.parameters or {},
                status="failed",
                error_message=str(exception),
                completed_at=datetime.now(timezone.utc),
                execution_time_ms=execution_time,
            )
            
            self.task_repo.save(task)
            return task
            
        except Exception as e:
            # If we can't save to database, at least log the error
            logger.error(
                f"Failed to create failed task record: {str(e)}",
                extra={
                    "task_type": task_data.task_type,
                    "original_error": str(exception),
                    "database_error": True,
                },
                exc_info=True,
            )
            
            # Return a minimal task object
            return Task(
                task_type=task_data.task_type,
                input_data=task_data.input_data,
                parameters=task_data.parameters or {},
                status="failed",
                error_message=f"Multiple errors: {exception}; {e}",
            )
    
    def _execute_task_logic(self, task: Task, task_data: TaskCreate) -> Dict[str, Any]:
        """Execute the actual task logic with decision points."""
        try:
            # Get execution plan
            execution_plan = self.decision_engine.get_execution_plan(
                task.task_type, 
                task.parameters
            )
            
            self.tracer.record_decision(
                "execution_plan_selected",
                {
                    "plan_id": execution_plan["id"],
                    "reason": execution_plan["reason"],
                    "complexity": execution_plan["complexity"],
                    "step_count": len(execution_plan.get("steps", [])),
                },
            )
            
            # Validate execution plan
            self._validate_execution_plan(execution_plan, task)
            
            # Execute steps
            step_results = []
            for step_idx, step in enumerate(execution_plan["steps"]):
                step_result = self._execute_step(step, task, step_idx)
                step_results.append(step_result)
                
                # Stop if step failed and not configured to continue
                if not step_result["success"] and not step.get("continue_on_failure", False):
                    logger.warning(
                        "Step failed and not configured to continue",
                        extra={
                            "task_id": task.id,
                            "step_index": step_idx,
                            "step_name": step.get("name"),
                            "continue_on_failure": step.get("continue_on_failure", False),
                        }
                    )
                    break
            
            # Process final results
            return self._process_results(step_results, task)
            
        except (AppException, ValidationError):
            # Re-raise our custom exceptions
            raise
            
        except Exception as e:
            # Wrap other exceptions
            raise AppException(
                message=f"Task logic execution failed: {str(e)}",
                extra={
                    "task_id": task.id,
                    "task_type": task.task_type,
                    "original_exception_type": type(e).__name__,
                    "original_exception": str(e),
                }
            ) from e
    
    def _validate_execution_plan(self, execution_plan: Dict[str, Any], task: Task) -> None:
        """Validate the execution plan before execution."""
        if not execution_plan.get("steps"):
            raise ValidationError(
                message="Execution plan has no steps",
                field="execution_plan.steps",
                value=None,
                validation_rules={"required": True, "type": "list", "min_length": 1},
                extra={
                    "task_id": task.id,
                    "plan_id": execution_plan.get("id"),
                }
            )
    
    def _execute_step(self, step: Dict[str, Any], task: Task, step_index: int) -> Dict[str, Any]:
        """Execute a single step."""
        step_start = time.time()
        
        with self.tracer.start_trace(
            f"step_{step['name']}", 
            {
                "step_name": step["name"], 
                "step_type": step["type"],
                "step_index": step_index,
                "task_id": task.id,
            }
        ):
            try:
                self.tracer.record_event("step_started", {
                    "step_name": step["name"],
                    "step_index": step_index,
                })
                
                # Execute based on step type
                if step["type"] == "llm_call":
                    output = self._execute_llm_call(step, task)
                elif step["type"] == "decision_point":
                    output = self._evaluate_decision_point(step, task)
                elif step["type"] == "data_transform":
                    output = self._transform_data(step, task)
                else:
                    raise ValidationError(
                        message=f"Unknown step type: {step['type']}",
                        field=f"step.type",
                        value=step["type"],
                        validation_rules={
                            "allowed_values": ["llm_call", "decision_point", "data_transform"]
                        },
                        extra={
                            "step_name": step["name"],
                            "step_index": step_index,
                        }
                    )
                
                self.tracer.record_event(
                    "step_completed",
                    {
                        "step_name": step["name"],
                        "step_index": step_index,
                        "output_type": type(output).__name__,
                        "output_size": len(str(output)) if output else 0,
                    },
                )
                
                return {
                    "success": True,
                    "output": output,
                    "execution_time_ms": int((time.time() - step_start) * 1000),
                    "step_name": step["name"],
                    "step_type": step["type"],
                }
                
            except (AppException, ValidationError) as e:
                self.tracer.record_error(
                    "step_failed",
                    f"Step failed with {e.__class__.__name__}: {e.message}",
                    {
                        "step_name": step["name"], 
                        "step_type": step["type"],
                        "step_index": step_index,
                        "exception_extra": e.extra,
                    },
                    skip_logging=True,  # Exception already logs itself
                )
                
                return {
                    "success": False,
                    "error": e.message,
                    "error_type": e.__class__.__name__,
                    "execution_time_ms": int((time.time() - step_start) * 1000),
                    "step_name": step["name"],
                    "step_type": step["type"],
                }
                
            except Exception as e:
                error_msg = f"Unexpected error in step execution: {str(e)}"
                self.tracer.record_error(
                    "step_failed",
                    error_msg,
                    {
                        "step_name": step["name"], 
                        "step_type": step["type"],
                        "step_index": step_index,
                        "original_exception_type": type(e).__name__,
                    },
                )
                
                return {
                    "success": False,
                    "error": error_msg,
                    "error_type": type(e).__name__,
                    "execution_time_ms": int((time.time() - step_start) * 1000),
                    "step_name": step["name"],
                    "step_type": step["type"],
                }
    
    def _execute_llm_call(self, step: Dict[str, Any], task: Task) -> Any:
        """Execute LLM call."""
        try:
            prompt = step["parameters"]["prompt"]
            llm_params = step["parameters"].get("llm_params", {})
            
            logger.debug("Executing LLM call", extra={
                "task_id": task.id,
                "prompt_preview": prompt[:100] if prompt else None,
                "llm_params": llm_params,
            })
            
            return self.llm_client.process(
                prompt,
                task.input_data,
                **llm_params,
            )
            
        except (AppException, ValidationError):
            raise  # Re-raise LLM client exceptions
        except Exception as e:
            raise AppException(
                message=f"LLM call execution failed: {str(e)}",
                extra={
                    "task_id": task.id,
                    "step_name": step["name"],
                    "prompt_preview": step["parameters"].get("prompt", "")[:100],
                    "original_exception_type": type(e).__name__,
                }
            ) from e
    
    def _evaluate_decision_point(self, step: Dict[str, Any], task: Task) -> bool:
        """Evaluate a decision point."""
        try:
            condition = step["parameters"]["condition"]
            
            logger.debug("Evaluating decision point", extra={
                "task_id": task.id,
                "condition": condition,
            })
            
            return self.decision_engine.evaluate_condition(
                condition, 
                task.input_data
            )
            
        except (AppException, ValidationError):
            raise  # Re-raise DecisionEngine exceptions
        except Exception as e:
            raise AppException(
                message=f"Decision point evaluation failed: {str(e)}",
                extra={
                    "task_id": task.id,
                    "step_name": step["name"],
                    "condition": step["parameters"].get("condition"),
                    "original_exception_type": type(e).__name__,
                }
            ) from e
    
    def _transform_data(self, step: Dict[str, Any], task: Task) -> Any:
        """Transform data."""
        try:
            transform_type = step["parameters"]["transform"]
            
            if transform_type == "extract_key_points":
                if isinstance(task.input_data, dict):
                    return list(task.input_data.keys())[:5]
                elif isinstance(task.input_data, list):
                    return task.input_data[:3]
                else:
                    raise ValidationError(
                        message="Cannot extract key points from non-dict/list input",
                        field="input_data",
                        value=type(task.input_data).__name__,
                        validation_rules={"type": ["dict", "list"]},
                        extra={"transform_type": transform_type}
                    )
            elif transform_type == "summarize":
                input_str = str(task.input_data)
                return f"Summary of {len(input_str)} characters"
            else:
                raise ValidationError(
                    message=f"Unknown transform type: {transform_type}",
                    field="transform",
                    value=transform_type,
                    validation_rules={"allowed_values": ["extract_key_points", "summarize"]},
                )
                
        except ValidationError:
            raise  # Re-raise validation errors
        except Exception as e:
            raise AppException(
                message=f"Data transformation failed: {str(e)}",
                extra={
                    "task_id": task.id,
                    "step_name": step["name"],
                    "transform_type": step["parameters"].get("transform"),
                    "original_exception_type": type(e).__name__,
                }
            ) from e
    
    def _process_results(self, step_results: List[Dict[str, Any]], task: Task) -> Dict[str, Any]:
        """Process step results into final output."""
        try:
            successful_steps = [r for r in step_results if r.get("success")]
            failed_steps = [r for r in step_results if not r.get("success")]
            
            if not failed_steps:
                return {
                    "success": True,
                    "output": {
                        "steps_completed": len(step_results),
                        "successful_steps": len(successful_steps),
                        "final_output": step_results[-1]["output"] if step_results else None,
                        "total_execution_time": sum(r.get("execution_time_ms", 0) for r in step_results),
                    }
                }
            else:
                return {
                    "success": False,
                    "error": f"{len(failed_steps)} step(s) failed",
                    "failed_steps": [
                        {
                            "step_name": r.get("step_name"),
                            "error": r.get("error"),
                            "error_type": r.get("error_type"),
                        }
                        for r in failed_steps
                    ],
                    "successful_steps": len(successful_steps),
                    "partial_results": [r.get("output") for r in successful_steps if r.get("output")],
                }
                
        except Exception as e:
            raise AppException(
                message=f"Failed to process step results: {str(e)}",
                extra={
                    "task_id": task.id,
                    "step_results_count": len(step_results),
                    "original_exception_type": type(e).__name__,
                }
            ) from e
    
    def get_task_by_id(self, task_id: int) -> Optional[Task]:
        """Get task by ID."""
        try:
            task = self.task_repo.get_by_id(task_id)
            if not task:
                logger.warning("Task not found", extra={"task_id": task_id})
            return task
        except Exception as e:
            raise AppException(
                message=f"Failed to retrieve task: {str(e)}",
                extra={
                    "task_id": task_id,
                    "original_exception_type": type(e).__name__,
                    "database_error": True,
                }
            ) from e
    
    def get_traces_for_task(self, task_id: int) -> Optional[List[ExecutionTrace]]:
        """Get all traces for a task."""
        try:
            task = self.task_repo.get_by_id(task_id)
            if not task:
                logger.warning("Task not found", extra={"task_id": task_id})
                return None
            return self.trace_repo.get_traces_for_task(task_id)
        except Exception as e:
            raise AppException(
                message=f"Failed to retrieve traces: {str(e)}",
                extra={
                    "task_id": task_id,
                    "original_exception_type": type(e).__name__,
                    "database_error": True,
                }
            ) from e
    
    def get_metrics_summary(self) -> Dict[str, Any]:
        """Get metrics summary."""
        try:
            return self.metrics.get_summary()
        except Exception as e:
            raise AppException(
                message=f"Failed to retrieve metrics: {str(e)}",
                extra={
                    "original_exception_type": type(e).__name__,
                    "metrics_error": True,
                }
            ) from e