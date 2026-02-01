from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

from app.extensions import db
from app.models.trace import ExecutionTrace


class TraceRepository:
    """Repository for ExecutionTrace model operations."""

    def save(self, trace: ExecutionTrace) -> ExecutionTrace:
        """Save trace to database."""
        db.session.add(trace)
        db.session.commit()
        return trace

    def get_by_id(self, trace_id: int) -> Optional[ExecutionTrace]:
        """Get trace by database ID."""
        return ExecutionTrace.query.get(trace_id)

    def get_by_trace_id(self, trace_uuid: str) -> Optional[ExecutionTrace]:
        """Get trace by UUID trace ID."""
        return ExecutionTrace.query.filter_by(trace_id=trace_uuid).first()

    def get_traces_for_task(self, task_id: int) -> List[ExecutionTrace]:
        """Get all traces for a task."""
        return (
            ExecutionTrace.query.filter_by(task_id=task_id)
            .order_by(ExecutionTrace.start_time)
            .all()
        )

    def get_trace_tree(self, root_trace_id: str) -> Dict[str, Any]:
        """Get a trace and all its child traces as a tree."""
        root_trace = self.get_by_trace_id(root_trace_id)
        if not root_trace:
            return {}

        # Get all traces with this root as ancestor
        all_traces = ExecutionTrace.query.filter(
            ExecutionTrace.trace_id.like(f"{root_trace_id}%")
        ).all()

        # Build tree structure
        trace_dict = {trace.trace_id: trace for trace in all_traces}
        tree = self._build_tree(root_trace_id, trace_dict)

        return tree

    def _build_tree(
        self, current_id: str, trace_dict: Dict[str, ExecutionTrace]
    ) -> Dict[str, Any]:
        """Recursively build trace tree."""
        trace = trace_dict.get(current_id)
        if not trace:
            return {}

        node = trace.to_dict()
        node["children"] = []

        # Find children
        for trace_id, child_trace in trace_dict.items():
            if child_trace.parent_trace_id == current_id:
                child_node = self._build_tree(trace_id, trace_dict)
                if child_node:
                    node["children"].append(child_node)

        return node

    def find_by_operation(
        self, operation: str, limit: int = 100
    ) -> List[ExecutionTrace]:
        """Find traces by operation type."""
        return (
            ExecutionTrace.query.filter_by(operation=operation)
            .order_by(ExecutionTrace.start_time.desc())
            .limit(limit)
            .all()
        )

    def find_failed_traces(
        self, hours: int = 24, limit: int = 100
    ) -> List[ExecutionTrace]:
        """Find failed traces within time period."""
        cutoff = datetime.now(timezone.utc) - timedelta(hours=hours)
        return (
            ExecutionTrace.query.filter(
                not ExecutionTrace.success and ExecutionTrace.created_at >= cutoff
            )
            .order_by(ExecutionTrace.created_at.desc())
            .limit(limit)
            .all()
        )

    def find_slow_traces(
        self, threshold_ms: int = 1000, hours: int = 24, limit: int = 100
    ) -> List[ExecutionTrace]:
        """Find slow traces exceeding threshold."""
        cutoff = datetime.now(timezone.utc) - timedelta(hours=hours)
        return (
            ExecutionTrace.query.filter(
                ExecutionTrace.duration_ms >= threshold_ms,
                ExecutionTrace.created_at >= cutoff,
            )
            .order_by(ExecutionTrace.duration_ms.desc())
            .limit(limit)
            .all()
        )

    def get_statistics(self, hours: int = 24) -> Dict[str, Any]:
        """Get trace statistics for a time period."""
        cutoff = datetime.now(timezone.utc) - timedelta(hours=hours)

        traces = ExecutionTrace.query.filter(ExecutionTrace.created_at >= cutoff).all()

        if not traces:
            return {
                "total": 0,
                "success_rate": 0,
                "by_operation": {},
                "avg_duration": 0,
                "error_count": 0,
            }

        by_operation = {}
        total_duration = 0
        successful_traces = 0
        error_count = 0

        for trace in traces:
            # Count by operation
            by_operation[trace.operation] = by_operation.get(trace.operation, 0) + 1

            # Calculate average duration
            if trace.duration_ms:
                total_duration += trace.duration_ms

            # Count successes
            if trace.success:
                successful_traces += 1

            # Count errors
            if trace.error:
                error_count += 1

        avg_duration = total_duration / len(traces) if traces else 0
        success_rate = successful_traces / len(traces) if traces else 0

        return {
            "total": len(traces),
            "success_rate": success_rate,
            "by_operation": by_operation,
            "avg_duration": avg_duration,
            "error_count": error_count,
        }

    def cleanup_old_traces(self, days: int = 30) -> int:
        """Clean up traces older than specified days."""
        cutoff = datetime.now(timezone.utc) - timedelta(days=days)
        deleted_count = ExecutionTrace.query.filter(
            ExecutionTrace.created_at < cutoff
        ).delete()
        db.session.commit()
        return deleted_count
