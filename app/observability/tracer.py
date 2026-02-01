import threading
import time
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from app.exceptions.base import AppException  # Import AppException
from app.models.trace import ExecutionTrace
from app.repositories.trace_repository import TraceRepository
from app.utils.logger import get_logger

logger = get_logger(__name__)


class TraceContext:
    """Thread-local trace context."""
    
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


class Tracer:
    def __init__(self):
        self.trace_repo = TraceRepository()
        self.context = TraceContext()
    
    @contextmanager
    def start_trace(self, operation: str, context: Optional[Dict[str, Any]] = None):
        trace_id = str(uuid.uuid4())
        start_time = time.time()
        
        parent_trace = self.context.current
        parent_trace_id = parent_trace.get("trace_id") if parent_trace else None
        
        trace_context = {
            "trace_id": trace_id,
            "parent_trace_id": parent_trace_id,
            "operation": operation,
            "context": context or {},
            "start_time": datetime.now(timezone.utc),
            "thread_id": threading.get_ident(),
            "decisions": [],
            "events": [],
            "error": None,
            "success": True,
        }
        
        self.context.push(trace_context)
        
        logger.debug("Trace started", extra={
            "trace_id": trace_id,
            "operation": operation,
            "context": context or {},
        })
        
        try:
            yield trace_context
            
        except AppException as e:
            # AppException already logs itself, but we still want to record it in trace
            self.record_error(
                error_type=e.__class__.__name__,
                error_message=e.message,
                context={
                    "exception_extra": e.extra,
                    "trace_id": trace_id,
                    "operation": operation,
                },
                skip_logging=True  # Don't log again, AppException already did
            )
            raise
            
        except Exception as e:
            # For non-AppException, log and wrap
            error_message = f"Trace operation '{operation}' failed: {str(e)}"
            
            # Record in trace
            self.record_error(
                error_type=type(e).__name__,
                error_message=error_message,
                context={
                    "original_exception": str(e),
                    "trace_id": trace_id,
                    "operation": operation,
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
                }
            ) from e
            
        finally:
            completed_trace = self.context.pop()
            if completed_trace:
                completed_trace["end_time"] = datetime.now(timezone.utc)
                completed_trace["duration_ms"] = int((time.time() - start_time) * 1000)
                
                # Persist if important operation
                if self._should_persist_trace(operation):
                    try:
                        self._persist_trace(completed_trace)
                    except Exception as e:
                        # Use AppException for persistence errors
                        error_msg = f"Failed to persist trace {trace_id}"
                        logger.error(error_msg, extra={
                            "trace_id": trace_id,
                            "error": str(e),
                            "error_type": type(e).__name__,
                        })
                        # Don't raise - tracing failure shouldn't break the application
                
                logger.debug(
                    "Trace completed",
                    extra={
                        "trace_id": trace_id,
                        "operation": operation,
                        "duration_ms": completed_trace["duration_ms"],
                        "success": completed_trace.get("success", True),
                    },
                )
    
    def record_decision(self, decision_type: str, context: Dict[str, Any]) -> None:
        """Record a decision in current trace."""
        current = self.context.current
        if current:
            if "decisions" not in current:
                current["decisions"] = []
            
            current["decisions"].append({
                "type": decision_type,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "context": context,
            })
        
        logger.info(
            "Decision recorded",
            extra={
                "decision_type": decision_type,
                "context": context,
                "trace_id": current.get("trace_id") if current else None,
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
            })
        
        logger.debug(
            "Event recorded",
            extra={
                "event_type": event_type,
                "data": data,
                "trace_id": current.get("trace_id") if current else None,
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
            }
            current["success"] = False
        
        if not skip_logging:
            logger.error(
                "Error recorded in trace",
                extra={
                    "error_type": error_type,
                    "error_message": error_message,
                    "context": context or {},
                    "trace_id": current.get("trace_id") if current else None,
                },
            )
    
    def get_current_traces(self) -> List[Dict[str, Any]]:
        """Get all traces from current context."""
        return self.context.stack.copy()
    
    def _should_persist_trace(self, operation: str) -> bool:
        """Determine if a trace should be persisted to database."""
        important_operations = ["task_execution", "llm_call", "decision_point"]
        return operation in important_operations
    
    def _persist_trace(self, trace_data: Dict[str, Any]) -> None:
        """Persist trace to database."""
        try:
            trace = ExecutionTrace(
                trace_id=trace_data["trace_id"],
                parent_trace_id=trace_data.get("parent_trace_id"),
                operation=trace_data["operation"],
                context=trace_data.get("context", {}),
                decisions=trace_data.get("decisions", []),
                events=trace_data.get("events", []),
                error=trace_data.get("error"),
                start_time=trace_data["start_time"],
                end_time=trace_data.get("end_time"),
                duration_ms=trace_data.get("duration_ms", 0),
                success=trace_data.get("success", True),
            )
            
            # Try to associate with task if possible
            task_id = trace_data.get("task_id")
            if task_id:
                trace.task_id = task_id
            
            self.trace_repo.save(trace)
            
        except Exception as e:
            # Use AppException for database persistence errors
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