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
    """Thread-local trace context with enhanced agent tracking."""
    
    def __init__(self):
        self._local = threading.local()
    
    @property
    def stack(self) -> List[Dict[str, Any]]:
        if not hasattr(self._local, 'stack'):
            self._local.stack = []
        return self._local.stack
    
    @property
    def current(self) -> Optional[Dict[str, Any]]:
        stack = self.stack
        return stack[-1] if stack else None
    
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
        """Get current agent context."""
        if not hasattr(self._local, 'agent_context'):
            self._local.agent_context = {}
        return self._local.agent_context
    
    @agent_context.setter
    def agent_context(self, value: Dict[str, Any]) -> None:
        self._local.agent_context = value
    
    def update_agent_context(self, updates: Dict[str, Any]) -> None:
        """Update agent context with new values."""
        self.agent_context.update(updates)


class Tracer:
    def __init__(self):
        self.trace_repo = TraceRepository()
        self.context = TraceContext()
        self._quality_hooks: List[Callable] = []
        self._agent_observation_hooks: List[Callable] = []
    
    @contextmanager
    def start_trace(self, operation: str, context: Optional[Dict[str, Any]] = None):
        trace_id = str(uuid.uuid4())
        start_time = time.time()
        
        parent_trace = self.context.current
        parent_trace_id = parent_trace.get("trace_id") if parent_trace else None
        
        # Initialize agent context if not present
        if "agent_context" not in self.context.agent_context:
            self.context.agent_context = {
                "agent_id": None,
                "agent_type": None,
                "goal": None,
                "step": 0,
                "total_steps": None,
            }
        
        trace_context = {
            "trace_id": trace_id,
            "parent_trace_id": parent_trace_id,
            "operation": operation,
            "context": context or {},
            "start_time": datetime.now(timezone.utc),
            "thread_id": threading.get_ident(),
            "decisions": [],
            "events": [],
            "agent_observations": [],  # Agent self-observations
            "quality_metrics": {},     # Quality metrics
            "error": None,
            "success": True,
            "agent_context": self.context.agent_context.copy(),  # Include current agent context
        }
        
        self.context.push(trace_context)
        
        logger.debug("Trace started", extra={
            "trace_id": trace_id,
            "operation": operation,
            "context": context or {},
            "agent_id": self.context.agent_context.get("agent_id"),
        })
        
        try:
            yield trace_context
            
        except AppException as e:
            # Record error in trace
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
            
            # Record in trace
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
            
            # Wrap in AppException with trace context
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
                
                # Calculate quality score if not already present
                if "quality_score" not in completed_trace.get("quality_metrics", {}):
                    self._calculate_trace_quality(completed_trace)
                
                # Run quality hooks
                for hook in self._quality_hooks:
                    try:
                        hook(completed_trace)
                    except Exception as e:
                        logger.error(f"Quality hook failed: {str(e)}", extra={
                            "trace_id": trace_id,
                            "hook_error": True,
                        })
                
                # Persist if important operation
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
                        "quality_score": completed_trace.get("quality_metrics", {}).get("quality_score"),
                        "agent_id": self.context.agent_context.get("agent_id"),
                    },
                )
    
    def record_decision(self, decision_type: str, context: Dict[str, Any]) -> None:
        """Record a decision in current trace."""
        current = self.context.current
        if current:
            if "decisions" not in current:
                current["decisions"] = []
            
            decision_record = {
                "type": decision_type,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "context": context,
                "agent_context": self.context.agent_context.copy(),
                "step": self.context.agent_context.get("step", 0),
            }
            
            # Add quality assessment for decision
            decision_quality = self._assess_decision_quality(decision_type, context)
            decision_record["quality"] = decision_quality
            
            current["decisions"].append(decision_record)
        
        logger.info(
            "Decision recorded",
            extra={
                "decision_type": decision_type,
                "context": self._sanitize_for_logging(context),
                "trace_id": current.get("trace_id") if current else None,
                "agent_id": self.context.agent_context.get("agent_id"),
                "decision_quality": decision_quality.get("overall_score") if current else None,
            },
        )
    
    def record_agent_observation(self, observation_type: str, data: Dict[str, Any]) -> None:
        """
        Record an agent's self-observation or thought process.
        
        Args:
            observation_type: Type of observation (thought, evaluation, etc.)
            data: Observation data
        """
        current = self.context.current
        if current:
            if "agent_observations" not in current:
                current["agent_observations"] = []
            
            observation = {
                "type": observation_type,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "data": data,
                "agent_context": self.context.agent_context.copy(),
                "step": self.context.agent_context.get("step", 0),
            }
            
            current["agent_observations"].append(observation)
            
            # Run observation hooks
            for hook in self._agent_observation_hooks:
                try:
                    hook(observation_type, data)
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
        """Record an event in current trace."""
        current = self.context.current
        if current:
            if "events" not in current:
                current["events"] = []
            
            current["events"].append({
                "type": event_type,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "data": data,
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
        """
        Record a quality metric for the current trace.
        
        Args:
            metric_name: Name of the quality metric
            value: Value of the metric
            metadata: Additional metadata about the metric
        """
        current = self.context.current
        if current:
            if "quality_metrics" not in current:
                current["quality_metrics"] = {}
            
            current["quality_metrics"][metric_name] = {
                "value": value,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "metadata": metadata or {},
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
        """Record an error in current trace."""
        current = self.context.current
        if current:
            current["error"] = {
                "type": error_type,
                "message": error_message,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "context": context or {},
                "agent_context": self.context.agent_context.copy(),
            }
            current["success"] = False
            
            # Record error as quality metric
            self.record_quality_metric("error_occurred", True, {
                "error_type": error_type,
                "error_message": error_message[:200],  # Truncate for safety
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
        """
        Update the current agent context.
        
        Args:
            **kwargs: Key-value pairs to update in agent context
        """
        self.context.update_agent_context(kwargs)
    
    def increment_agent_step(self) -> int:
        """Increment the agent step counter and return new value."""
        current_step = self.context.agent_context.get("step", 0)
        new_step = current_step + 1
        self.update_agent_context(step=new_step)
        return new_step
    
    def add_quality_hook(self, hook: Callable) -> None:
        """Add a hook to be called after trace completion for quality assessment."""
        self._quality_hooks.append(hook)
    
    def add_agent_observation_hook(self, hook: Callable) -> None:
        """Add a hook to be called when agent observations are recorded."""
        self._agent_observation_hooks.append(hook)
    
    def get_current_traces(self) -> List[Dict[str, Any]]:
        """Get all traces from current context."""
        return self.context.stack.copy()
    
    def get_current_agent_context(self) -> Dict[str, Any]:
        """Get current agent context."""
        return self.context.agent_context.copy()
    
    def _should_persist_trace(self, operation: str) -> bool:
        """Determine if a trace should be persisted to database."""
        important_operations = [
            "task_execution", "llm_call", "decision_point",
            "agent_decision", "self_evaluation", "quality_assessment"
        ]
        return operation in important_operations

    def _persist_trace(self, trace_data: Dict[str, Any]) -> None:
        """Persist trace to database."""
        try:
            # Ensure end_time is timezone-aware
            end_time = trace_data.get("end_time")
            if end_time and end_time.tzinfo is None:
                end_time = end_time.replace(tzinfo=timezone.utc)
                
            trace = ExecutionTrace(
                trace_id=trace_data["trace_id"],
                parent_trace_id=trace_data.get("parent_trace_id"),
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
            )
            
            # Try to associate with task if possible
            task_id = trace_data.get("task_id")
            if task_id:
                trace.task_id = task_id
            
            # Store agent context if available
            agent_context = trace_data.get("agent_context")
            if agent_context:
                trace.agent_context = agent_context
            
            self.trace_repo.save(trace)
            
        except Exception as e:
            raise AppException(
                message=f"Failed to persist trace {trace_data.get('trace_id')} to database",
                extra={
                    "trace_id": trace_data.get("trace_id"),
                    "error_type": type(e).__name__,
                    "error_details": str(e),
                    "operation": trace_data.get("operation"),
                    "persistence_error": True,
                }
            )
    
    def _calculate_trace_quality(self, trace_data: Dict[str, Any]) -> None:
        """Calculate quality metrics for a trace."""
        quality_metrics = trace_data.get("quality_metrics", {})
        
        # Calculate based on various factors
        scores = []
        
        # Success factor
        if trace_data.get("success", False):
            scores.append(1.0)
        else:
            scores.append(0.0)
        
        # Duration factor (faster is better, but not too fast)
        duration = trace_data.get("duration_ms", 0)
        if duration > 0:
            # Ideal duration is between 100ms and 5000ms
            if 100 <= duration <= 5000:
                scores.append(1.0)
            elif duration < 100:
                scores.append(0.7)  # Possibly too fast
            elif duration > 10000:
                scores.append(0.3)  # Too slow
            else:
                scores.append(0.5)  # Acceptable
        
        # Decision quality factor
        decisions = trace_data.get("decisions", [])
        if decisions:
            decision_scores = [d.get("quality", {}).get("overall_score", 0.5) for d in decisions]
            avg_decision_score = sum(decision_scores) / len(decision_scores)
            scores.append(avg_decision_score)
        
        # Calculate overall quality score
        if scores:
            overall_score = sum(scores) / len(scores)
            quality_metrics["quality_score"] = overall_score
            quality_metrics["quality_factors"] = {
                "success": scores[0] if len(scores) > 0 else 0.5,
                "duration": scores[1] if len(scores) > 1 else 0.5,
                "decision_quality": scores[2] if len(scores) > 2 else 0.5,
            }
        
        trace_data["quality_metrics"] = quality_metrics
    
    def _assess_decision_quality(self, decision_type: str, context: Dict) -> Dict[str, Any]:
        """Assess the quality of a decision."""
        quality = {
            "has_context": bool(context),
            "context_size": len(str(context)) if context else 0,
            "decision_type_specificity": self._assess_specificity(decision_type),
        }
        
        # Calculate overall score
        factors = [
            1.0 if quality["has_context"] else 0.0,
            min(1.0, quality["context_size"] / 100),  # More context is better
            quality["decision_type_specificity"],
        ]
        
        quality["overall_score"] = sum(factors) / len(factors)
        quality["factor_weights"] = ["context_presence", "context_detail", "specificity"]
        
        return quality
    
    def _assess_specificity(self, decision_type: str) -> float:
        """Assess how specific a decision type is."""
        generic_types = ["decision", "choice", "selection"]
        
        if any(generic in decision_type.lower() for generic in generic_types):
            return 0.5  # Generic
        elif "." in decision_type or "_" in decision_type:
            return 0.9  # Specific
        else:
            return 0.7  # Moderately specific
    
    def _sanitize_for_logging(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Sanitize data for logging (remove sensitive info)."""
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
                    for item in value[:3]  # Limit list size for logging
                ]
            elif isinstance(value, str) and len(value) > 200:
                sanitized[key] = value[:200] + "..."
            else:
                sanitized[key] = value
        
        return sanitized
