import threading
import time
from collections import defaultdict, deque
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List

from app.utils.logger import get_logger

logger = get_logger(__name__)


class MetricsCollector:
    """Collects and aggregates performance metrics."""
    
    def __init__(self, retention_period_hours: int = 24):
        self.retention_period = timedelta(hours=retention_period_hours)
        self.metrics = {
            "task_executions": deque(maxlen=10000),
            "llm_calls": deque(maxlen=10000),
            "decisions": deque(maxlen=10000),
            "errors": deque(maxlen=10000),
        }
        self._cleanup_interval = 3600  # Cleanup every hour
        self._last_cleanup = time.time()
        self._lock = threading.RLock()  # Thread safety
    
    def record_task_completion(
        self, task_type: str, status: str, execution_time_ms: int
    ):
        """Record task completion metrics."""
        metric = {
            "timestamp": datetime.now(timezone.utc),
            "task_type": task_type,
            "status": status,
            "execution_time_ms": execution_time_ms,
        }
        
        with self._lock:
            self.metrics["task_executions"].append(metric)
        
        logger.debug("Task metric recorded", extra=metric)
        self._perform_cleanup_if_needed()
    
    def record_llm_call(
        self,
        model: str,
        prompt_tokens: int,
        completion_tokens: int,
        latency_ms: int,
        success: bool,
    ):
        """Record LLM call metrics."""
        metric = {
            "timestamp": datetime.now(timezone.utc),
            "model": model,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
            "latency_ms": latency_ms,
            "success": success,
        }
        
        with self._lock:
            self.metrics["llm_calls"].append(metric)
        
        logger.debug("LLM metric recorded", extra=metric)
        self._perform_cleanup_if_needed()
    
    def record_decision(
        self, decision_type: str, processing_time_ms: int, context_size: int
    ):
        """Record decision processing metrics."""
        metric = {
            "timestamp": datetime.now(timezone.utc),
            "decision_type": decision_type,
            "processing_time_ms": processing_time_ms,
            "context_size": context_size,
        }
        
        with self._lock:
            self.metrics["decisions"].append(metric)
        
        logger.debug("Decision metric recorded", extra=metric)
        self._perform_cleanup_if_needed()
    
    def get_summary(self) -> Dict[str, Any]:
        """Get summary of all metrics."""
        now = datetime.now(timezone.utc)
        hour_ago = now - timedelta(hours=1)
        day_ago = now - timedelta(days=1)
        
        with self._lock:
            return {
                "task_metrics": self._summarize_tasks(hour_ago, day_ago),
                "llm_metrics": self._summarize_llm_calls(hour_ago, day_ago),
                "decision_metrics": self._summarize_decisions(hour_ago, day_ago),
                "error_metrics": self._summarize_errors(hour_ago, day_ago),
                "timestamp": now.isoformat(),
            }
    
    def _summarize_tasks(self, hour_ago: datetime, day_ago: datetime) -> Dict[str, Any]:
        """Summarize task execution metrics."""
        with self._lock:
            tasks = list(self.metrics["task_executions"])
        
        recent_tasks = [t for t in tasks if t["timestamp"] >= hour_ago]
        daily_tasks = [t for t in tasks if t["timestamp"] >= day_ago]
        
        return {
            "last_hour": {
                "total": len(recent_tasks),
                "by_type": self._count_by_type(recent_tasks, "task_type"),
                "by_status": self._count_by_type(recent_tasks, "status"),
                "avg_execution_time": self._average_execution_time(recent_tasks),
            },
            "last_24_hours": {
                "total": len(daily_tasks),
                "by_type": self._count_by_type(daily_tasks, "task_type"),
                "by_status": self._count_by_type(daily_tasks, "status"),
                "avg_execution_time": self._average_execution_time(daily_tasks),
            },
        }
    
    def _summarize_llm_calls(
        self, hour_ago: datetime, day_ago: datetime
    ) -> Dict[str, Any]:
        """Summarize LLM call metrics."""
        with self._lock:
            calls = list(self.metrics["llm_calls"])
        
        recent_calls = [c for c in calls if c["timestamp"] >= hour_ago]
        daily_calls = [c for c in calls if c["timestamp"] >= day_ago]
        
        return {
            "last_hour": {
                "total": len(recent_calls),
                "success_rate": self._success_rate(recent_calls),
                "avg_latency": self._average_latency(recent_calls),
                "total_tokens": sum(c["total_tokens"] for c in recent_calls),
            },
            "last_24_hours": {
                "total": len(daily_calls),
                "success_rate": self._success_rate(daily_calls),
                "avg_latency": self._average_latency(daily_calls),
                "total_tokens": sum(c["total_tokens"] for c in daily_calls),
            },
        }
    
    def _summarize_decisions(
        self, hour_ago: datetime, day_ago: datetime
    ) -> Dict[str, Any]:
        """Summarize decision metrics."""
        with self._lock:
            decisions = list(self.metrics["decisions"])
        
        recent_decisions = [d for d in decisions if d["timestamp"] >= hour_ago]
        daily_decisions = [d for d in decisions if d["timestamp"] >= day_ago]
        
        return {
            "last_hour": {
                "total": len(recent_decisions),
                "avg_processing_time": self._average_processing_time(recent_decisions),
                "by_type": self._count_by_type(recent_decisions, "decision_type"),
            },
            "last_24_hours": {
                "total": len(daily_decisions),
                "avg_processing_time": self._average_processing_time(daily_decisions),
                "by_type": self._count_by_type(daily_decisions, "decision_type"),
            },
        }
    
    def _summarize_errors(
        self, hour_ago: datetime, day_ago: datetime
    ) -> Dict[str, Any]:
        """Summarize error metrics."""
        with self._lock:
            errors = list(self.metrics["errors"])
        
        recent_errors = [e for e in errors if e["timestamp"] >= hour_ago]
        daily_errors = [e for e in errors if e["timestamp"] >= day_ago]
        
        return {
            "last_hour": {
                "total": len(recent_errors),
                "by_type": self._count_by_type(recent_errors, "error_type"),
                "by_component": self._count_by_type(recent_errors, "component"),
            },
            "last_24_hours": {
                "total": len(daily_errors),
                "by_type": self._count_by_type(daily_errors, "error_type"),
                "by_component": self._count_by_type(daily_errors, "component"),
            },
        }
    
    def _count_by_type(self, items: List[Dict[str, Any]], key: str) -> Dict[str, int]:
        """Count items by a specific key."""
        counts = defaultdict(int)
        for item in items:
            counts[item.get(key, "unknown")] += 1
        return dict(counts)
    
    def _average_execution_time(self, tasks: List[Dict[str, Any]]) -> float:
        """Calculate average execution time."""
        if not tasks:
            return 0.0
        return sum(t["execution_time_ms"] for t in tasks) / len(tasks)
    
    def _average_latency(self, calls: List[Dict[str, Any]]) -> float:
        """Calculate average latency."""
        if not calls:
            return 0.0
        return sum(c["latency_ms"] for c in calls) / len(calls)
    
    def _average_processing_time(self, decisions: List[Dict[str, Any]]) -> float:
        """Calculate average processing time."""
        if not decisions:
            return 0.0
        return sum(d["processing_time_ms"] for d in decisions) / len(decisions)
    
    def _success_rate(self, items: List[Dict[str, Any]]) -> float:
        """Calculate success rate."""
        if not items:
            return 0.0
        successful = sum(1 for item in items if item.get("success", False))
        return successful / len(items)
    
    def _perform_cleanup_if_needed(self):
        """Clean up old metrics if cleanup interval has passed."""
        current_time = time.time()
        if current_time - self._last_cleanup >= self._cleanup_interval:
            self._cleanup_old_metrics()
            self._last_cleanup = current_time
    
    def _cleanup_old_metrics(self):
        """Remove metrics older than retention period."""
        cutoff = datetime.now(timezone.utc) - self.retention_period
        
        with self._lock:
            for metric_type in self.metrics:
                # Filter out old metrics
                items = list(self.metrics[metric_type])
                filtered_items = [item for item in items if item["timestamp"] >= cutoff]
                self.metrics[metric_type] = deque(filtered_items, maxlen=10000)
