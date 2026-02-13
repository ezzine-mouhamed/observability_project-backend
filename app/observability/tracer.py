import threading
import time
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Callable

from app.exceptions.base import AppException
from app.models.trace import ExecutionTrace
from app.repositories.trace_repository import TraceRepository
from app.utils.logger import get_logger

logger = get_logger(__name__)


class TraceContext:
    def __init__(self):
        self._local = threading.local()
        self._parent_map = {}
    
    @property
    def stack(self) -> List[Dict[str, Any]]:
        if not hasattr(self._local, 'stack'):
            self._local.stack = []
        return self._local.stack

    @property
    def current(self) -> Optional[Dict[str, Any]]:
        stack = self.stack
        return stack[-1] if stack else None
    
    def set_parent(self, child_trace_id: str, parent_trace_id: str) -> None:
        self._parent_map[child_trace_id] = parent_trace_id

    def push(self, trace_context: Dict[str, Any]) -> None:
        self.stack.append(trace_context)
    
    def pop(self) -> Optional[Dict[str, Any]]:
        if self.stack:
            return self.stack.pop()
        return None
    
    def clear(self) -> None:
        if hasattr(self._local, 'stack'):
            self._local.stack = []
    
    @property
    def agent_context(self) -> Dict[str, Any]:
        if not hasattr(self._local, 'agent_context'):
            self._local.agent_context = {}
        return self._local.agent_context
    
    @agent_context.setter
    def agent_context(self, value: Dict[str, Any]) -> None:
        self._local.agent_context = value
    
    def update_agent_context(self, updates: Dict[str, Any]) -> None:
        self.agent_context.update(updates)

class Tracer:
    def __init__(self):
        self.trace_repo = TraceRepository()
        self.context = TraceContext()
        self._quality_hooks: List[Callable] = []
        self._agent_observation_hooks: List[Callable] = []
        # In-memory expected durations (could be loaded from DB)
        self._expected_durations = {
            "summarize": 5000,
            "translate": 3000,
            "classify": 2000,
            "analyze": 10000,
            "extract": 1500,
        }
    
    @contextmanager
    def start_trace(self, operation: str, context: Optional[Dict[str, Any]] = None, 
                    parent_trace_id: Optional[str] = None, 
                    agent_context: Optional[Dict[str, Any]] = None):
        trace_id = str(uuid.uuid4())
        start_time = time.time()
        
        # Determine parent trace ID
        if parent_trace_id is None:
            parent_trace = self.context.current
            parent_trace_id = parent_trace.get("trace_id") if parent_trace else None
        else:
            parent_trace = None  # we don't have the parent object if ID is given explicitly
        
        if parent_trace_id:
            self.context.set_parent(trace_id, parent_trace_id)
        
        # Merge context: start with parent's context (if any), then update with provided context
        merged_context = {}
        if parent_trace and parent_trace.get("context"):
            merged_context.update(parent_trace["context"])
        if context:
            merged_context.update(context)
        
        # Use explicit agent_context if provided, otherwise copy current
        if agent_context is None:
            agent_context = self.context.agent_context.copy()
        
        # Initialize agent context if this is root trace
        if self.context.current is None and not self.context.agent_context.get("agent_id"):
            self.context.agent_context = {
                "agent_id": None,
                "agent_type": None,
                "goal": None,
                "step": 0,
                "total_steps": None,
            }
            agent_context = self.context.agent_context.copy()

        trace_context = {
            "trace_id": trace_id,
            "parent_trace_id": parent_trace_id,
            "operation": operation,
            "context": merged_context,
            "start_time": datetime.now(timezone.utc),
            "thread_id": threading.get_ident(),
            "decisions": [],
            "events": [],
            "agent_observations": [],
            "quality_metrics": {},
            "error": None,
            "success": True,
            "agent_context": agent_context.copy(),
        }
        
        self.context.push(trace_context)
        
        logger.debug("Trace started", extra={
            "trace_id": trace_id,
            "operation": operation,
            "parent_trace_id": parent_trace_id,
            "context": merged_context,
            "agent_id": self.context.agent_context.get("agent_id"),
        })
        
        try:
            yield trace_context
            
        except AppException as e:
            self.record_error(
                error_type=e.__class__.__name__,
                error_message=e.message,
                context={
                    "exception_extra": e.extra,
                    "trace_id": trace_id,
                    "operation": operation,
                    "agent_context": self.context.agent_context,
                },
                skip_logging=True
            )
            raise
            
        except Exception as e:
            error_message = f"Trace operation '{operation}' failed: {str(e)}"
            
            self.record_error(
                error_type=type(e).__name__,
                error_message=error_message,
                context={
                    "original_exception": str(e),
                    "trace_id": trace_id,
                    "operation": operation,
                    "agent_context": self.context.agent_context,
                }
            )
            
            raise AppException(
                message=error_message,
                extra={
                    "trace_id": trace_id,
                    "operation": operation,
                    "original_exception_type": type(e).__name__,
                    "original_exception": str(e),
                    "tracer_error": True,
                    "agent_context": self.context.agent_context,
                }
            ) from e
            
        finally:
            completed_trace = self.context.pop()
            if completed_trace:
                completed_trace["end_time"] = datetime.now(timezone.utc)
                completed_trace["duration_ms"] = int((time.time() - start_time) * 1000)
                
                if "quality_score" not in completed_trace.get("quality_metrics", {}):
                    self._calculate_trace_quality(completed_trace)
                
                for hook in self._quality_hooks:
                    try:
                        hook(completed_trace.copy())
                    except Exception as e:
                        logger.error(f"Quality hook failed: {str(e)}", extra={
                            "trace_id": trace_id,
                            "hook_error": True,
                        })
                
                if self._should_persist_trace(operation):
                    try:
                        self._persist_trace(completed_trace)
                    except Exception as e:
                        error_msg = f"Failed to persist trace {trace_id}"
                        logger.error(error_msg, extra={
                            "trace_id": trace_id,
                            "error": str(e),
                            "error_type": type(e).__name__,
                        })
                
                logger.debug(
                    "Trace completed",
                    extra={
                        "trace_id": trace_id,
                        "operation": operation,
                        "duration_ms": completed_trace["duration_ms"],
                        "success": completed_trace.get("success", True),
                        "quality_score": completed_trace.get("quality_metrics", {}).get("composite_quality_score"),
                        "agent_id": self.context.agent_context.get("agent_id"),
                    },
                )

    def record_decision(self, decision_type: str, context: Dict[str, Any]) -> None:
        current = self.context.current
        if current:
            if "decisions" not in current:
                current["decisions"] = []
            
            decision_record = {
                "type": decision_type,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "context": context.copy(),
                "agent_context": self.context.agent_context.copy(),
                "step": self.context.agent_context.get("step", 0),
            }
            
            decision_record["quality"] = {
                "overall_score": None,
                "status": "pending",
                "measurement_method": "pending"
            }
            
            current["decisions"].append(decision_record)
        
        logger.info(
            "Decision recorded",
            extra={
                "decision_type": decision_type,
                "trace_id": current.get("trace_id") if current else None,
                "agent_id": self.context.agent_context.get("agent_id"),
            },
        )

    def record_agent_observation(self, observation_type: str, data: Dict[str, Any]) -> None:
        current = self.context.current
        if current:
            if "agent_observations" not in current:
                current["agent_observations"] = []
            
            observation = {
                "type": observation_type,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "data": data.copy(),
                "agent_context": self.context.agent_context.copy(),
                "step": self.context.agent_context.get("step", 0),
            }
            
            current["agent_observations"].append(observation)
            
            for hook in self._agent_observation_hooks:
                try:
                    hook(observation_type, data.copy())
                except Exception as e:
                    logger.error(f"Agent observation hook failed: {str(e)}")
        
        logger.debug(
            "Agent observation recorded",
            extra={
                "observation_type": observation_type,
                "trace_id": current.get("trace_id") if current else None,
                "agent_id": self.context.agent_context.get("agent_id"),
            },
        )
    
    def record_event(self, event_type: str, data: Dict[str, Any]) -> None:
        current = self.context.current
        if current:
            if "events" not in current:
                current["events"] = []
            
            current["events"].append({
                "type": event_type,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "data": data.copy(),
                "agent_context": self.context.agent_context.copy(),
            })
        
        logger.debug(
            "Event recorded",
            extra={
                "event_type": event_type,
                "data": self._sanitize_for_logging(data),
                "trace_id": current.get("trace_id") if current else None,
                "agent_id": self.context.agent_context.get("agent_id"),
            },
        )
    
    def record_quality_metric(self, metric_name: str, value: Any, metadata: Optional[Dict] = None) -> None:
        current = self.context.current
        if current:
            if "quality_metrics" not in current:
                current["quality_metrics"] = {}
            
            current["quality_metrics"][metric_name] = {
                "value": value,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "metadata": metadata.copy() if metadata else {},
            }
        
        logger.debug(
            "Quality metric recorded",
            extra={
                "metric_name": metric_name,
                "value": value,
                "trace_id": current.get("trace_id") if current else None,
                "agent_id": self.context.agent_context.get("agent_id"),
            },
        )
    
    def record_error(
        self,
        error_type: str,
        error_message: str,
        context: Optional[Dict[str, Any]] = None,
        skip_logging: bool = False,
    ) -> None:
        current = self.context.current
        if current:
            current["error"] = {
                "type": error_type,
                "message": error_message,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "context": context.copy() if context else {},
                "agent_context": self.context.agent_context.copy(),
            }
            current["success"] = False
            
            self.record_quality_metric("error_occurred", True, {
                "error_type": error_type,
                "error_message": error_message[:200],
            })
        
        if not skip_logging:
            logger.error(
                "Error recorded in trace",
                extra={
                    "error_type": error_type,
                    "error_message": error_message,
                    "context": context or {},
                    "trace_id": current.get("trace_id") if current else None,
                    "agent_id": self.context.agent_context.get("agent_id"),
                },
            )
    
    def update_agent_context(self, **kwargs) -> None:
        self.context.update_agent_context(kwargs)
        current = self.context.current
        if current and "agent_context" in current:
            current["agent_context"].update(kwargs)
    
    def increment_agent_step(self) -> int:
        current_step = self.context.agent_context.get("step", 0)
        new_step = current_step + 1
        self.update_agent_context(step=new_step)
        return new_step
    
    def add_quality_hook(self, hook: Callable) -> None:
        self._quality_hooks.append(hook)
    
    def add_agent_observation_hook(self, hook: Callable) -> None:
        self._agent_observation_hooks.append(hook)
    
    def get_current_traces(self) -> List[Dict[str, Any]]:
        return [trace.copy() for trace in self.context.stack]
    
    def get_current_agent_context(self) -> Dict[str, Any]:
        return self.context.agent_context.copy()
    
    def _should_persist_trace(self, operation: str) -> bool:
        irrelevant_operations = [
            "condition_evaluation", "step_result_formatting",
            "step_summarize_processing", "step_input_validation"
        ]
        return operation not in irrelevant_operations

    def _persist_trace(self, trace_data: Dict[str, Any]) -> None:
        try:
            if "parent_trace_id" not in trace_data or trace_data["parent_trace_id"] is None:
                trace_data["parent_trace_id"] = self.context._parent_map.get(trace_data["trace_id"])
            
            end_time = trace_data.get("end_time")
            if end_time and end_time.tzinfo is None:
                end_time = end_time.replace(tzinfo=timezone.utc)
                
            trace = ExecutionTrace(
                trace_id=trace_data["trace_id"],
                parent_trace_id=trace_data.get("parent_trace_id"),
                task_id=trace_data.get("context", {}).get("task_id"),
                operation=trace_data["operation"],
                context=trace_data.get("context", {}),
                decisions=trace_data.get("decisions", []),
                events=trace_data.get("events", []),
                agent_observations=trace_data.get("agent_observations", []),
                quality_metrics=trace_data.get("quality_metrics", {}),
                error=trace_data.get("error"),
                start_time=trace_data["start_time"],
                end_time=end_time,
                duration_ms=trace_data.get("duration_ms", 0),
                success=trace_data.get("success", True),
                agent_context=trace_data.get("agent_context", {}),
            )
            
            self.trace_repo.save(trace)
            
        except Exception as e:
            raise AppException(
                message=f"Failed to persist trace {trace_data.get('trace_id')} to database",
                extra={
                    "trace_id": trace_data.get("trace_id"),
                    "parent_trace_id": trace_data.get("parent_trace_id"),
                    "error_type": type(e).__name__,
                    "error_details": str(e),
                    "operation": trace_data.get("operation"),
                    "persistence_error": True,
                }
            )

    def verify_trace_hierarchy(self, trace_id: str) -> Dict[str, Any]:
        trace = self.trace_repo.get_by_trace_id(trace_id)
        if not trace:
            return {"error": "Trace not found"}
        
        hierarchy = {
            "trace": {
                "id": trace.trace_id,
                "operation": trace.operation,
                "parent": trace.parent_trace_id,
            }
        }
        
        children = ExecutionTrace.query.filter_by(parent_trace_id=trace_id).all()
        if children:
            hierarchy["children"] = [
                {
                    "id": child.trace_id,
                    "operation": child.operation,
                    "depth": 1,
                }
                for child in children
            ]
        
        ancestors = []
        current_trace = trace
        while current_trace.parent_trace_id:
            parent = self.trace_repo.get_by_trace_id(current_trace.parent_trace_id)
            if parent:
                ancestors.append({
                    "id": parent.trace_id,
                    "operation": parent.operation,
                    "depth": len(ancestors) + 1,
                })
                current_trace = parent
            else:
                break
        
        if ancestors:
            hierarchy["ancestors"] = ancestors
        
        return hierarchy
    
    def _calculate_trace_quality(self, trace_data: Dict[str, Any]) -> None:
        quality_metrics = {}
        
        success_factor = 1.0 if trace_data.get("success", False) else 0.0
        quality_metrics["success_factor"] = success_factor
        
        duration = trace_data.get("duration_ms", 0)
        if duration > 0:
            if duration <= 5000:
                efficiency = 1.0
            elif duration <= 30000:
                efficiency = 1.0 - ((duration - 5000) / 25000 * 0.5)
            else:
                efficiency = max(0.0, 0.5 - ((duration - 30000) / 60000 * 0.5))
            quality_metrics["efficiency_factor"] = efficiency
        else:
            quality_metrics["efficiency_factor"] = 0.5
        
        decisions = trace_data.get("decisions", [])
        if decisions:
            decision_scores = []
            for decision in decisions:
                if isinstance(decision, dict) and "quality" in decision:
                    score = decision["quality"].get("overall_score")
                    if score is not None:
                        decision_scores.append(score)
            
            if decision_scores:
                quality_metrics["decision_quality_factor"] = sum(decision_scores) / len(decision_scores)
            else:
                quality_metrics["decision_quality_factor"] = 0.5
        else:
            quality_metrics["decision_quality_factor"] = 0.5
        
        weights = {
            "success_factor": 0.4,
            "efficiency_factor": 0.3,
            "decision_quality_factor": 0.3
        }
        
        composite_score = 0.0
        for factor, weight in weights.items():
            if factor in quality_metrics:
                composite_score += quality_metrics[factor] * weight
        
        quality_metrics["composite_quality_score"] = composite_score
        trace_data["quality_metrics"] = quality_metrics

    def _get_expected_duration(self, task_type: Optional[str]) -> int:
        return self._expected_durations.get(task_type, 5000)

    def update_decision_quality_with_outcome(self, trace_id: str, task_status: str, task_quality: float, execution_time_ms: int, task_type: str = None) -> None:
        """Update the decision quality in the in‑memory trace (will be persisted later)."""
        # Find the trace in the current context stack
        trace = None
        for t in self.context.stack:
            if t["trace_id"] == trace_id:
                trace = t
                break
        if not trace or not trace.get("decisions"):
            return

        updated = False
        for decision in trace["decisions"]:
            if decision.get("type") == "execution_plan_selected":
                success_score = 1.0 if task_status == "completed" else 0.0
                quality_score = task_quality if task_quality else 0.5
                expected = self._get_expected_duration(task_type)
                efficiency = min(1.0, expected / execution_time_ms) if execution_time_ms and execution_time_ms > 0 else 0.5

                final_score = (
                    success_score * 0.5 +
                    quality_score * 0.3 +
                    efficiency * 0.2
                )

                decision["quality"] = {
                    "overall_score": final_score,
                    "success_score": success_score,
                    "output_quality": quality_score,
                    "efficiency": efficiency,
                    "task_status": task_status,
                    "execution_time_ms": execution_time_ms,
                    "measurement_method": "outcome_based"
                }
                updated = True
                break

        if updated:
            logger.info(f"Updated decision quality for trace {trace_id} to {final_score:.2f}")

    def _assess_specificity(self, decision_type: str) -> float:
        generic_types = ["decision", "choice", "selection"]
        
        if any(generic in decision_type.lower() for generic in generic_types):
            return 0.5
        elif "." in decision_type or "_" in decision_type:
            return 0.9
        else:
            return 0.7

    def _sanitize_for_logging(self, data: Dict[str, Any]) -> Dict[str, Any]:
        if not data:
            return data
        
        sanitized = {}
        sensitive_keys = ["password", "secret", "key", "token", "api_key", "auth"]
        
        for key, value in data.items():
            key_lower = key.lower()
            if any(sensitive in key_lower for sensitive in sensitive_keys):
                sanitized[key] = "***REDACTED***"
            elif isinstance(value, dict):
                sanitized[key] = self._sanitize_for_logging(value)
            elif isinstance(value, list):
                sanitized[key] = [
                    self._sanitize_for_logging(item) if isinstance(item, dict) else 
                    ("***REDACTED***" if isinstance(item, str) and any(sensitive in item.lower() for sensitive in sensitive_keys) else item)
                    for item in value[:3]
                ]
            elif isinstance(value, str) and len(value) > 200:
                sanitized[key] = value[:200] + "..."
            else:
                sanitized[key] = value
        
        return sanitized
