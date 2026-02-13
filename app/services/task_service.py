import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from app.exceptions.base import AppException
from app.exceptions.validation import ValidationError
from app.models.task import Task
from app.models.trace import ExecutionTrace
from app.observability.agent_observer import AgentObserver
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
    def __init__(self, tracer: Optional[Tracer] = None):
        self.tracer = tracer or Tracer()
        self.task_repo = TaskRepository()
        self.trace_repo = TraceRepository()
        
        self.agent_observer = AgentObserver(self.tracer)
        self.llm_client = LLMClient(tracer=self.tracer)
        
        self.decision_engine = DecisionEngine(
            tracer=self.tracer,
            agent_observer=self.agent_observer,
            llm_client=self.llm_client
        )
        
        self.tracer.add_quality_hook(self._on_trace_quality_assessed)
        self.tracer.add_agent_observation_hook(self._on_agent_observation)

    def execute_task(self, task_data: TaskCreate) -> TaskResult:
        start_time = time.time()
        task = None
        current_trace_id = None

        try:
            self._validate_task_data(task_data)
            task = self._create_task_record(task_data)

            self.tracer.update_agent_context(
                agent_id=f"task_executor_{task.id}",
                agent_type="task_processor",
                goal=f"Execute {task.task_type} task",
                step=0,
                total_steps=None
            )

            with self.tracer.start_trace(
                "task_execution",
                {
                    "task_id": task.id,
                    "task_type": task.task_type,
                    "input_size": len(str(task.input_data)) if task.input_data else 0,
                }
            ) as current_trace:
                current_trace_id = current_trace["trace_id"]
                
                self.tracer.context.current["task_id"] = task.id

                task.start_execution()
                self.task_repo.save(task)
                logger.info("Task execution started", extra={"task_id": task.id})

                logic_outcome = self._execute_task_logic(task)
                result = logic_outcome["result"]
                step_results = logic_outcome["step_results"]

                task.complete_execution(
                    success=result["success"],
                    output=result.get("output"),
                    error=result.get("error")
                )
                task.calculate_quality_score(result["success"])
                self.task_repo.save(task)

                self._record_self_evaluation(task, step_results, result)

                self.tracer.update_decision_quality_with_outcome(
                    trace_id=current_trace_id,
                    task_status=task.status,
                    task_quality=task.quality_score,
                    execution_time_ms=task.execution_time_ms,
                    task_type=task.task_type
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

                traces = self.tracer.get_current_traces()

                if result["success"]:
                    learning_results = self.learn_from_execution(task, traces)
                    logger.info(
                        "Learning from execution completed",
                        extra={
                            "task_id": task.id,
                            "insights_generated": learning_results.get("insights_generated", 0),
                            "insights_applied": learning_results.get("insights_applied", 0),
                        }
                    )
                return TaskResult(
                    task=task,
                    success=result["success"],
                    traces=traces,
                )
                
        except (AppException, ValidationError) as e:
            logger.error(
                f"Task execution failed with {e.__class__.__name__}: {e.message}",
                extra={"task_type": task_data.task_type, **e.extra},
            )
            
            if task:
                task = self._update_task_to_failed(task, start_time, e)
            else:
                task = self._handle_task_failure(task_data, start_time, e)
                
            return TaskResult(task=task, success=False, traces=self.tracer.get_current_traces())
            
        except Exception as e:
            logger.error(
                f"Unexpected error in task execution: {str(e)}",
                extra={
                    "task_type": task_data.task_type,
                    "original_exception_type": type(e).__name__,
                },
                exc_info=True,
            )
            
            wrapped_exception = AppException(
                message=f"Unexpected error during task execution: {str(e)}",
                extra={
                    "task_type": task_data.task_type,
                    "original_exception_type": type(e).__name__,
                    "original_exception": str(e),
                    "unexpected_error": True,
                }
            )
            
            if task:
                task = self._update_task_to_failed(task, start_time, wrapped_exception)
            else:
                task = self._handle_task_failure(task_data, start_time, wrapped_exception)
                
            return TaskResult(task=task, success=False, traces=self.tracer.get_current_traces())

    def _update_task_to_failed(self, task: Task, start_time: float, exception: Exception) -> Task:
        try:
            execution_time = int((time.time() - start_time) * 1000)
            
            task.status = "failed"
            task.error_message = str(exception)
            task.completed_at = datetime.now(timezone.utc)
            
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
            
            task.calculate_quality_score(False)
            
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
        if not task_data.task_type:
            raise ValidationError(
                message="Task type is required",
                field="task_type",
                value=task_data.task_type,
                validation_rules={"required": True, "type": "string"},
                log=False,
            )
        
        if task_data.input_data and len(str(task_data.input_data)) > 10000:
            raise ValidationError(
                message="Input data too large (max 10000 characters)",
                field="input_data",
                value=f"{len(str(task_data.input_data))} characters",
                validation_rules={"max_length": 10000},
                log=False,
            )
    
    def _create_task_record(self, task_data: TaskCreate) -> Task:
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
        try:
            execution_time = int((time.time() - start_time) * 1000)
            
            task = Task(
                task_type=task_data.task_type,
                input_data=task_data.input_data,
                parameters=task_data.parameters or {},
                status="failed",
                error_message=str(exception),
                completed_at=datetime.now(timezone.utc),
                execution_time_ms=execution_time,
            )
            
            task.calculate_quality_score(False)
            
            self.task_repo.save(task)
            return task
            
        except Exception as e:
            logger.error(
                f"Failed to create failed task record: {str(e)}",
                extra={
                    "task_type": task_data.task_type,
                    "original_error": str(exception),
                    "database_error": True,
                },
                exc_info=True,
            )
            
            return Task(
                task_type=task_data.task_type,
                input_data=task_data.input_data,
                parameters=task_data.parameters or {},
                status="failed",
                error_message=f"Multiple errors: {exception}; {e}",
            )

    def _execute_task_logic(self, task: Task) -> Dict[str, Any]:
        try:
            execution_plan = self.decision_engine.get_execution_plan(
                task.task_type,
                task.parameters
            )
            print("Execution plan received:", execution_plan)

            step_count = len(execution_plan.get("steps", []))
            self.tracer.update_agent_context(total_steps=step_count)

            self.tracer.record_decision(
                "execution_plan_selected",
                {
                    "plan_id": execution_plan["id"],
                    "reason": execution_plan["reason"],
                    "complexity": execution_plan["complexity"],
                    "step_count": step_count,
                },
            )
            
            self._validate_execution_plan(execution_plan, task)
            
            step_results = []
            current_data = task.input_data
            
            for step_idx, step in enumerate(execution_plan["steps"]):
                logger.debug(
                    f"Executing step {step_idx}: {step['name']} with current data preview",
                    extra={
                        "step_name": step["name"],
                        "current_data_preview": str(current_data)[:100] if current_data else None,
                    }
                )
                step_result = self._execute_step(step, task, step_idx, current_data)
                logger.debug(
                    f"Step {step_idx} result: {step_result}",
                    extra={
                        "step_name": step["name"],
                        "step_result": step_result,
                    }
                )
                step_results.append(step_result)
                
                if step_result["success"] and step_result.get("output") is not None:
                    current_data = step_result["output"]
                
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
            
            result = self._process_results(step_results, task, current_data)
            self._record_step_performance_metrics(task, step_results, result)
            
            return {"result": result, "step_results": step_results}
            
        except (AppException, ValidationError):
            raise
        except Exception as e:
            raise AppException(
                message=f"Task logic execution failed: {str(e)}",
                extra={
                    "task_id": task.id,
                    "task_type": task.task_type,
                    "original_exception_type": type(e).__name__,
                    "original_exception": str(e),
                }
            ) from e

    def _record_step_performance_metrics(self, task: Task, step_results: List[Dict[str, Any]], result: Dict[str, Any]) -> None:
        try:
            for step_result in step_results:
                step_name = step_result.get("step_name", f"step_{step_results.index(step_result)}")
                execution_time = step_result.get("execution_time_ms")
                if execution_time is None:
                    execution_time = 0
                task.record_performance_metric(
                    f"{step_name}_execution_time_ms",
                    execution_time
                )
            
            total_time = 0
            successful_steps = 0
            for r in step_results:
                exec_time = r.get("execution_time_ms")
                if exec_time is not None:
                    total_time += exec_time
                if r.get("success"):
                    successful_steps += 1
            
            task.record_performance_metric("total_execution_time_ms", total_time)
            task.record_performance_metric("successful_steps_count", successful_steps)
            task.record_performance_metric("total_steps_count", len(step_results))
            
            success_rate = successful_steps / len(step_results) if step_results else 0.0
            task.record_performance_metric("step_success_rate", success_rate)
            
            if result.get("success") and result.get("output"):
                output = result["output"]
                if isinstance(output, dict) and "final_output" in output:
                    final_output = output["final_output"]
                    if isinstance(final_output, str):
                        task.record_performance_metric("final_output_length", len(final_output))
                        task.record_performance_metric("final_output_word_count", len(final_output.split()))
            
        except Exception as e:
            logger.error(
                f"Failed to record performance metrics: {str(e)}",
                extra={"task_id": task.id},
                exc_info=True,
            )

    def _validate_execution_plan(self, execution_plan: Dict[str, Any], task: Task) -> None:
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

    def _execute_step(self, step: Dict[str, Any], task: Task, step_index: int, current_data: Any = None) -> Dict[str, Any]:
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
                
                data_to_process = current_data if current_data is not None else task.input_data
                
                if step["type"] == "llm_call":
                    output = self._execute_llm_call(step, task, data_to_process)
                elif step["type"] == "decision_point":
                    output = self._evaluate_decision_point(step, task, data_to_process)
                elif step["type"] == "data_transform":
                    output = self._transform_data(step, task, data_to_process)
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
                    skip_logging=True,
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

    def _execute_llm_call(self, step: Dict[str, Any], task: Task, data: Any) -> Any:
        try:
            prompt = step["parameters"]["prompt"]
            llm_params = step["parameters"].get("llm_params", {})
            
            logger.debug(
                f"Executing LLM call - task_id: {task.id}, input_preview: {str(data)[:100] if data else None}",
                extra={"llm_params": llm_params}
            )
            
            return self.llm_client.process(
                prompt,
                data,
                **llm_params,
            )
            
        except (AppException, ValidationError):
            raise
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

    def _evaluate_decision_point(self, step: Dict[str, Any], task: Task, data: Any) -> bool:
        try:
            condition = step["parameters"]["condition"]
            
            logger.debug("Evaluating decision point", extra={
                "task_id": task.id,
                "condition": condition,
            })
            
            result = self.decision_engine.evaluate_condition(
                condition,
                task,
                data
            )
            
            task.record_validation(
                check_name=condition,
                passed=result["result"],
                details={
                    "context": result.get("content", {}),
                    "condition": condition,
                    "step_name": step["name"]
                }
            )
            
            return result["content"]
            
        except (AppException, ValidationError) as e:
            task.record_validation(
                check_name=condition,
                passed=False,
                details={
                    "error": e.message,
                    "condition": condition,
                    "step_name": step["name"]
                }
            )
            raise
        except Exception as e:
            task.record_validation(
                check_name=condition,
                passed=False,
                details={
                    "error": str(e),
                    "condition": condition,
                    "step_name": step["name"]
                }
            )
            raise AppException(
                message=f"Decision point evaluation failed: {str(e)}",
                extra={
                    "task_id": task.id,
                    "step_name": step["name"],
                    "condition": step["parameters"].get("condition"),
                    "original_exception_type": type(e).__name__,
                }
            ) from e

    def _transform_data(self, step: Dict[str, Any], task: Task, data: Any) -> Any:
        try:
            transform_type = step["parameters"]["transform"]
            
            if transform_type == "extract_key_points":
                if isinstance(data, dict):
                    return list(data.keys())[:5]
                elif isinstance(data, list):
                    return data[:3]
                else:
                    raise ValidationError(
                        message="Cannot extract key points from non-dict/list input",
                        field="input_data",
                        value=type(data).__name__,
                        validation_rules={"type": ["dict", "list"]},
                        extra={"transform_type": transform_type}
                    )
            elif transform_type == "summarize":
                if isinstance(data, dict) and "content" in data:
                    summary = data["content"]
                else:
                    summary = str(data)
                
                return summary
            else:
                raise ValidationError(
                    message=f"Unknown transform type: {transform_type}",
                    field="transform",
                    value=transform_type,
                    validation_rules={"allowed_values": ["extract_key_points", "summarize"]},
                )
                
        except ValidationError:
            raise
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
    
    def _process_results(self, step_results: List[Dict[str, Any]], task: Task, final_data: Any = None) -> Dict[str, Any]:
        try:
            successful_steps = [r for r in step_results if r.get("success")]
            failed_steps = [r for r in step_results if not r.get("success")]
            
            if not failed_steps:
                if final_data is not None:
                    final_output = final_data
                elif step_results:
                    final_output = step_results[-1]["output"]
                else:
                    final_output = None
                
                total_time = 0
                for r in step_results:
                    exec_time = r.get("execution_time_ms")
                    if exec_time is not None:
                        total_time += exec_time
                
                return {
                    "success": True,
                    "output": {
                        "steps_completed": len(step_results),
                        "successful_steps": len(successful_steps),
                        "final_output": final_output,
                        "total_execution_time": total_time,
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

    def _record_self_evaluation(self, task: Task, step_results: List[Dict[str, Any]], result: Dict[str, Any]) -> None:
        if not step_results:
            return
        completeness = len([s for s in step_results if s.get("success")]) / len(step_results)
        appropriateness = 1.0 if result["success"] else 0.0
        expected = self.tracer._get_expected_duration(task.task_type)
        efficiency = min(1.0, expected / task.execution_time_ms) if task.execution_time_ms and task.execution_time_ms > 0 else 0.5
        
        self.agent_observer.record_self_evaluation(
            agent_name=f"task_executor_{task.id}",
            task_id=str(task.id),
            evaluation_criteria={
                "completeness": 0.3,
                "appropriateness": 0.4,
                "efficiency": 0.3,
            },
            self_scores={
                "completeness": completeness,
                "appropriateness": appropriateness,
                "efficiency": efficiency,
            },
            justification="Self-evaluation based on execution results",
            improvements_suggested=[],
            metadata={"step_count": len(step_results), "success": result["success"]}
        )
    
    def get_task_by_id(self, task_id: int) -> Optional[Task]:
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

    def _on_trace_quality_assessed(self, trace_data: Dict[str, Any]) -> None:
        try:
            quality_score = trace_data.get("quality_metrics", {}).get("composite_quality_score", 0)
            operation = trace_data.get("operation", "unknown")
            trace_id = trace_data.get("trace_id", "unknown")
            
            if quality_score < 0.5:
                logger.warning(
                    f"Low quality trace detected: {trace_id}",
                    extra={
                        "trace_id": trace_id,
                        "operation": operation,
                        "quality_score": quality_score,
                        "trigger": "quality_hook",
                    }
                )
                
                agent_name = trace_data.get("agent_context", {}).get("agent_id", "unknown")
                if self.agent_observer and agent_name != "unknown":
                    self.agent_observer.detect_behavior_pattern(
                        agent_name=agent_name,
                        behavior_type="low_quality_operation",
                        pattern_data={
                            "operation": operation,
                            "quality_score": quality_score,
                            "trace_id": trace_id,
                        },
                        frequency=1,
                        context={"hook_triggered": True}
                    )
            
            elif quality_score > 0.9:
                logger.info(
                    f"High quality trace: {trace_id}",
                    extra={
                        "trace_id": trace_id,
                        "operation": operation,
                        "quality_score": quality_score,
                        "exemplar": True,
                    }
                )
                
        except Exception as e:
            logger.error(
                f"Quality hook failed: {str(e)}",
                extra={
                    "trace_id": trace_data.get("trace_id", "unknown"),
                    "hook_error": True,
                    "error_type": type(e).__name__,
                }
            )

    def _on_agent_observation(self, observation_type: str, data: Dict[str, Any]) -> None:
        try:
            agent_name = data.get("agent_name", data.get("agent_id", "unknown"))
            observation_id = data.get("observation_id", data.get("id", "unknown"))
            
            if observation_type == "thought_process":
                thought_chain_length = data.get("chain_length", 0)
                has_conclusion = data.get("has_conclusion", False)
                
                if thought_chain_length > 10 and not has_conclusion:
                    logger.warning(
                        f"Agent {agent_name} has long thought chain without conclusion",
                        extra={
                            "agent_name": agent_name,
                            "observation_id": observation_id,
                            "thought_chain_length": thought_chain_length,
                            "has_conclusion": has_conclusion,
                            "pattern": "inefficient_thinking",
                        }
                    )
            
            elif observation_type == "decision_rationale":
                confidence = data.get("confidence_score", 0)
                if confidence < 0.3:
                    logger.warning(
                        f"Agent {agent_name} made low-confidence decision",
                        extra={
                            "agent_name": agent_name,
                            "observation_id": observation_id,
                            "confidence_score": confidence,
                            "decision_id": data.get("decision_id", "unknown"),
                            "pattern": "low_confidence",
                        }
                    )
            
            elif observation_type == "self_evaluation":
                overall_score = data.get("overall_score", 0)
                task_id = data.get("task_id", "unknown")
                
                if overall_score < 0.4:
                    logger.info(
                        f"Agent {agent_name} gave harsh self-evaluation",
                        extra={
                            "agent_name": agent_name,
                            "observation_id": observation_id,
                            "overall_score": overall_score,
                            "task_id": task_id,
                            "self_critical": True,
                        }
                    )
            
        except Exception as e:
            logger.error(
                f"Agent observation hook failed: {str(e)}",
                extra={
                    "observation_type": observation_type,
                    "data_sample": str(data)[:200] if data else "none",
                    "hook_error": True,
                    "error_type": type(e).__name__,
                }
            )

    def learn_from_execution(self, task: Task, traces: List[ExecutionTrace]) -> Dict[str, Any]:
        try:
            learning_start = time.time()
            
            with self.tracer.start_trace(
                "agent_learning",
                {
                    "task_id": task.id,
                    "task_type": task.task_type,
                    "trace_count": len(traces),
                    "quality_score": task.quality_score,
                }
            ):
                agents_involved = self._extract_agents_from_traces(traces)
                
                if not agents_involved:
                    logger.warning("No agents found in traces for learning", extra={"task_id": task.id})
                    return {"learning_applied": False, "reason": "no_agents_found"}
                
                learning_results = {
                    "task_id": task.id,
                    "agents_analyzed": agents_involved,
                    "insights_generated": 0,
                    "insights_applied": 0,
                    "behavior_updates": [],
                    "quality_improvements": [],
                    "learning_duration_ms": 0,
                }
                
                for agent_name in agents_involved:
                    agent_learning = self._learn_for_agent(agent_name, task, traces)
                    learning_results["insights_generated"] += agent_learning.get("insights_generated", 0)
                    learning_results["insights_applied"] += agent_learning.get("insights_applied", 0)
                    
                    if agent_learning.get("behavior_updates"):
                        learning_results["behavior_updates"].extend(agent_learning["behavior_updates"])
                    
                    if agent_learning.get("quality_improvements"):
                        learning_results["quality_improvements"].extend(agent_learning["quality_improvements"])
                
                self._update_task_with_learnings(task, learning_results)
                
                learning_duration = int((time.time() - learning_start) * 1000)
                learning_results["learning_duration_ms"] = learning_duration
                
                self.tracer.record_event(
                    "learning_completed",
                    {
                        "task_id": task.id,
                        "insights_generated": learning_results["insights_generated"],
                        "insights_applied": learning_results["insights_applied"],
                        "learning_duration_ms": learning_duration,
                        "agents_count": len(agents_involved),
                    }
                )
                
                logger.info(
                    "Agent learning completed",
                    extra={
                        "task_id": task.id,
                        "insights_generated": learning_results["insights_generated"],
                        "insights_applied": learning_results["insights_applied"],
                        "agents": agents_involved,
                    }
                )
                
                return learning_results
                
        except Exception as e:
            logger.error(
                "Failed to learn from execution",
                extra={
                    "task_id": task.id,
                    "error": str(e),
                    "error_type": type(e).__name__,
                    "learning_failed": True,
                }
            )
            return {"learning_applied": False, "error": str(e), "error_type": type(e).__name__}

    def _extract_agents_from_traces(self, traces: List[Any]) -> List[str]:
        agents = set()
        
        for trace in traces:
            if hasattr(trace, 'agent_context'):
                if trace.agent_context and trace.agent_context.get("agent_id"):
                    agents.add(trace.agent_context["agent_id"])
            else:
                if trace.get("agent_context", {}).get("agent_id"):
                    agents.add(trace["agent_context"]["agent_id"])
            
            decisions = trace.decisions if hasattr(trace, 'decisions') else trace.get("decisions", [])
            for decision in decisions:
                if isinstance(decision, dict):
                    if decision.get("agent_context", {}).get("agent_id"):
                        agents.add(decision["agent_context"]["agent_id"])
        
        return list(agents)

    def _learn_for_agent(self, agent_name: str, task: Task, traces: List[ExecutionTrace]) -> Dict[str, Any]:
        try:
            agent_learning = {
                "agent_name": agent_name,
                "insights_generated": 0,
                "insights_applied": 0,
                "behavior_updates": [],
                "quality_improvements": [],
            }
            
            insights = self.agent_observer.generate_insights_from_observations(
                agent_name=agent_name,
                time_window_hours=24
            )
            
            agent_learning["insights_generated"] = len(insights)
            
            if not insights:
                logger.debug(f"No insights generated for agent {agent_name} - insufficient data")
                return agent_learning
            
            applicable_insights = self._prioritize_insights(insights, task)
            
            for insight in applicable_insights:
                try:
                    application_result = self._apply_agent_insight(agent_name, insight, task, traces)
                    
                    if application_result.get("applied", False):
                        agent_learning["insights_applied"] += 1
                        
                        if application_result.get("behavior_update"):
                            agent_learning["behavior_updates"].append(
                                application_result["behavior_update"]
                            )
                        
                        if application_result.get("quality_improvement"):
                            agent_learning["quality_improvements"].append(
                                application_result["quality_improvement"]
                            )
                        
                        logger.info(
                            f"Applied insight for agent {agent_name}",
                            extra={
                                "insight_type": insight.get("insight_type"),
                                "application_result": application_result,
                                "task_id": task.id,
                            }
                        )
                    
                except Exception as e:
                    logger.error(
                        f"Failed to apply insight for agent {agent_name}",
                        extra={
                            "agent_name": agent_name,
                            "insight_type": insight.get("insight_type"),
                            "error": str(e),
                            "task_id": task.id,
                        }
                    )
            
            return agent_learning
            
        except Exception as e:
            logger.error(
                f"Failed to learn for agent {agent_name}",
                extra={
                    "agent_name": agent_name,
                    "task_id": task.id,
                    "error": str(e),
                    "agent_learning_failed": True,
                }
            )
            return {"agent_name": agent_name, "error": str(e)}

    def _prioritize_insights(self, insights: List[Dict[str, Any]], task: Task) -> List[Dict[str, Any]]:
        if not insights:
            return []
        
        scored_insights = []
        for insight in insights:
            score = 0
            
            insight_type = insight.get("insight_type", "")
            if insight_type in ["quality_improvement", "efficiency_gain"]:
                score += 3
            elif insight_type in ["decision_improvement", "error_pattern"]:
                score += 2
            else:
                score += 1
            
            confidence = insight.get("confidence_score", 0)
            score += confidence * 2
            
            if task.quality_score and task.quality_score < 0.7:
                if insight_type == "quality_improvement":
                    score += 2
            
            impact = insight.get("impact_prediction", "low")
            if impact == "high":
                score += 3
            elif impact == "medium":
                score += 2
            else:
                score += 1
            
            scored_insights.append((score, insight))
        
        scored_insights.sort(key=lambda x: x[0], reverse=True)
        
        top_n = min(3, len(scored_insights))
        return [insight for _, insight in scored_insights[:top_n]]

    def _apply_agent_insight(
        self, 
        agent_name: str, 
        insight: Dict[str, Any], 
        task: Task, 
        traces: List[ExecutionTrace]
    ) -> Dict[str, Any]:
        insight_type = insight.get("insight_type", "")
        insight_data = insight.get("insight_data", {})
        
        application_result = {
            "applied": False,
            "insight_type": insight_type,
            "agent_name": agent_name,
            "application_method": "",
            "changes_made": [],
        }
        
        try:
            if insight_type == "quality_improvement":
                result = self._apply_quality_improvement(agent_name, insight_data, task, traces)
                application_result.update(result)
                application_result["application_method"] = "quality_threshold_adjustment"
            
            elif insight_type == "decision_improvement":
                result = self._apply_decision_improvement(agent_name, insight_data, task, traces)
                application_result.update(result)
                application_result["application_method"] = "decision_process_optimization"
            
            elif insight_type == "efficiency_gain":
                result = self._apply_efficiency_gain(agent_name, insight_data, task, traces)
                application_result.update(result)
                application_result["application_method"] = "execution_optimization"
            
            elif insight_type == "behavior_pattern":
                result = self._apply_behavior_pattern_insight(agent_name, insight_data, task, traces)
                application_result.update(result)
                application_result["application_method"] = "behavior_pattern_mitigation"
            
            else:
                result = self._apply_general_insight(agent_name, insight, task, traces)
                application_result.update(result)
                application_result["application_method"] = "general_optimization"
            
            if application_result["applied"]:
                self._record_insight_application(agent_name, insight, application_result)
            
            return application_result
            
        except Exception as e:
            logger.error(
                f"Failed to apply insight {insight_type} for agent {agent_name}",
                extra={
                    "agent_name": agent_name,
                    "insight_type": insight_type,
                    "error": str(e),
                    "application_failed": True,
                }
            )
            return application_result

    def _apply_quality_improvement(
        self, 
        agent_name: str, 
        insight_data: Dict[str, Any], 
        task: Task, 
        traces: List[ExecutionTrace]
    ) -> Dict[str, Any]:
        """
        Apply quality improvement insights.
        
        Example: Adjust quality thresholds based on performance.
        """
        current_quality = insight_data.get("current_quality", 0)
        threshold = insight_data.get("threshold", 0.7)
        gap = insight_data.get("gap", 0)
        
        changes_made = []
        
        if current_quality < threshold:
            # Do NOT modify the observer's thresholds – they are static configuration.
            # Instead, record the recommendation.
            changes_made.append({
                "change": f"Consider adjusting minimum quality threshold from {threshold} to {max(0.5, threshold - 0.1)}",
                "reason": f"Current quality {current_quality} below threshold",
                "agent_name": agent_name,
            })
            
            changes_made.append({
                "change": "Added recommendation for additional quality checks",
                "reason": "Persistent quality issues detected",
                "agent_name": agent_name,
            })
        
        return {
            "applied": len(changes_made) > 0,
            "behavior_update": f"Quality improvement suggestions recorded for {agent_name}",
            "quality_improvement": f"Target quality: {current_quality + gap}",
            "changes_made": changes_made,
        }
    def _apply_decision_improvement(
        self, 
        agent_name: str, 
        insight_data: Dict[str, Any], 
        task: Task, 
        traces: List[ExecutionTrace]
    ) -> Dict[str, Any]:
        current_decision_quality = insight_data.get("current_decision_quality", 0)
        target_quality = insight_data.get("target_quality", 0.7)
        gap = insight_data.get("gap", 0)
        
        changes_made = []
        
        if current_decision_quality < target_quality:
            changes_made.append({
                "change": "Increased minimum alternatives considered from 1 to 2",
                "reason": f"Decision quality {current_decision_quality} below target {target_quality}",
                "agent_name": agent_name,
            })
            
            changes_made.append({
                "change": "Added requirement for detailed rationale in decisions",
                "reason": "Improve decision traceability and quality",
                "agent_name": agent_name,
            })
        
        return {
            "applied": len(changes_made) > 0,
            "behavior_update": f"Decision process optimized for {agent_name}",
            "quality_improvement": f"Target decision quality: {target_quality}",
            "changes_made": changes_made,
        }

    def _apply_efficiency_gain(
        self, 
        agent_name: str, 
        insight_data: Dict[str, Any], 
        task: Task, 
        traces: List[ExecutionTrace]
    ) -> Dict[str, Any]:
        changes_made = []
        
        execution_times = []
        for trace in traces:
            if trace.duration_ms:
                execution_times.append(trace.duration_ms)
        
        if execution_times:
            avg_execution_time = sum(execution_times) / len(execution_times)
            
            if avg_execution_time > 5000:
                changes_made.append({
                    "change": "Added performance monitoring for slow operations",
                    "reason": f"Average execution time {avg_execution_time:.0f}ms is high",
                    "agent_name": agent_name,
                })
                
                changes_made.append({
                    "change": "Enabled result caching for repeated operations",
                    "reason": "Reduce redundant processing",
                    "agent_name": agent_name,
                })
        
        return {
            "applied": len(changes_made) > 0,
            "behavior_update": f"Efficiency optimizations applied for {agent_name}",
            "quality_improvement": "Expected performance improvement",
            "changes_made": changes_made,
        }

    def _apply_behavior_pattern_insight(
        self, 
        agent_name: str, 
        insight_data: Dict[str, Any], 
        task: Task, 
        traces: List[ExecutionTrace]
    ) -> Dict[str, Any]:
        pattern_type = insight_data.get("pattern_type", "")
        occurrence_count = insight_data.get("occurrence_count", 0)
        significance = insight_data.get("significance", "low")
        
        changes_made = []
        
        if significance == "high" and occurrence_count >= 3:
            changes_made.append({
                "change": f"Added monitoring for pattern: {pattern_type}",
                "reason": f"Pattern occurred {occurrence_count} times with high significance",
                "agent_name": agent_name,
            })
            
            changes_made.append({
                "change": "Triggered detailed analysis for pattern root cause",
                "reason": "Proactive behavior optimization",
                "agent_name": agent_name,
            })
        
        return {
            "applied": len(changes_made) > 0,
            "behavior_update": f"Behavior pattern mitigation for {agent_name}",
            "quality_improvement": "Reduced pattern recurrence",
            "changes_made": changes_made,
        }

    def _apply_general_insight(
        self, 
        agent_name: str, 
        insight: Dict[str, Any], 
        task: Task, 
        traces: List[ExecutionTrace]
    ) -> Dict[str, Any]:
        recommended_action = insight.get("recommended_action", "")
        
        changes_made = [{
            "change": f"Applied general insight: {recommended_action[:100]}...",
            "reason": "Insight-based optimization",
            "agent_name": agent_name,
        }]
        
        return {
            "applied": True,
            "behavior_update": f"General optimization applied for {agent_name}",
            "quality_improvement": "General performance improvement",
            "changes_made": changes_made,
        }

    def _record_insight_application(
        self, 
        agent_name: str, 
        insight: Dict[str, Any], 
        application_result: Dict[str, Any]
    ) -> None:
        try:
            insight_id = insight.get("insight_id")
            if insight_id and hasattr(self.agent_observer, '_insight_repository'):
                from app.models.agent_insight import AgentInsight
                
                db_insight = self.agent_observer._insight_repository.get_by_id(insight_id)
                if db_insight:
                    db_insight.mark_applied("applied_via_learning")
                    self.agent_observer._insight_repository.save(db_insight)
                    
                    logger.debug(
                        f"Insight {insight_id} marked as applied",
                        extra={
                            "agent_name": agent_name,
                            "insight_type": insight.get("insight_type"),
                            "application_result": application_result.get("application_method"),
                        }
                    )
        
        except Exception as e:
            logger.error(
                f"Failed to record insight application",
                extra={
                    "agent_name": agent_name,
                    "insight_id": insight.get("insight_id"),
                    "error": str(e),
                }
            )

    def _update_task_with_learnings(self, task: Task, learning_results: Dict[str, Any]) -> None:
        try:
            if not task.parameters:
                task.parameters = {}
            
            task.parameters["learning_metadata"] = {
                "learned_at": datetime.now(timezone.utc).isoformat(),
                "insights_generated": learning_results.get("insights_generated", 0),
                "insights_applied": learning_results.get("insights_applied", 0),
                "agents_analyzed": learning_results.get("agents_analyzed", []),
            }
            
            self.task_repo.save(task)
            
            logger.debug(
                "Task updated with learning metadata",
                extra={
                    "task_id": task.id,
                    "insights_generated": learning_results.get("insights_generated", 0),
                    "insights_applied": learning_results.get("insights_applied", 0),
                }
            )
        
        except Exception as e:
            logger.error(
                "Failed to update task with learnings",
                extra={
                    "task_id": task.id,
                    "error": str(e),
                }
            )
