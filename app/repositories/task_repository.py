from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

from app.extensions import db
from app.models.task import Task


class TaskRepository:
    """Repository for Task model operations."""

    def save(self, task: Task) -> Task:
        """Save task to database."""
        db.session.add(task)
        db.session.commit()
        return task

    def get_by_id(self, task_id: int) -> Optional[Task]:
        """Get task by ID."""
        return db.session.get(Task, task_id)

    def get_by_trace_id(self, trace_id: str) -> Optional[Task]:
        """Get task by trace ID."""
        from app.models.trace import ExecutionTrace

        trace = ExecutionTrace.query.filter_by(trace_id=trace_id).first()
        if trace and trace.task_id:
            return self.get_by_id(trace.task_id)
        return None

    def find_by_status(self, status: str, limit: int = 100) -> List[Task]:
        """Find tasks by status."""
        return (
            Task.query.filter_by(status=status)
            .order_by(Task.created_at.desc())
            .limit(limit)
            .all()
        )

    def find_by_type(self, task_type: str, limit: int = 100) -> List[Task]:
        """Find tasks by type."""
        return (
            Task.query.filter_by(task_type=task_type)
            .order_by(Task.created_at.desc())
            .limit(limit)
            .all()
        )

    def find_recent(self, hours: int = 24, limit: int = 100) -> List[Task]:
        """Find recent tasks."""
        cutoff = datetime.now(timezone.utc) - timedelta(hours=hours)
        return (
            Task.query.filter(Task.created_at >= cutoff)
            .order_by(Task.created_at.desc())
            .limit(limit)
            .all()
        )

    def update_status(
        self, task_id: int, status: str, error_message: Optional[str] = None
    ) -> Optional[Task]:
        """Update task status."""
        task = self.get_by_id(task_id)
        if task:
            task.status = status
            if error_message:
                task.error_message = error_message
            if status in ["completed", "failed"]:
                task.completed_at = datetime.now(timezone.utc)
            db.session.commit()
        return task

    def delete(self, task_id: int) -> bool:
        """Delete task by ID."""
        task = self.get_by_id(task_id)
        if task:
            db.session.delete(task)
            db.session.commit()
            return True
        return False

    def get_statistics(self, hours: int = 24) -> Dict[str, Any]:
        """Get task statistics for a time period."""
        cutoff = datetime.now(timezone.utc) - timedelta(hours=hours)

        tasks = Task.query.filter(Task.created_at >= cutoff).all()

        if not tasks:
            return {"total": 0, "by_status": {}, "by_type": {}, "avg_execution_time": 0}

        by_status = {}
        by_type = {}
        total_execution_time = 0
        completed_tasks = 0

        for task in tasks:
            # Count by status
            by_status[task.status] = by_status.get(task.status, 0) + 1

            # Count by type
            by_type[task.task_type] = by_type.get(task.task_type, 0) + 1

            # Calculate average execution time for completed tasks
            if task.execution_time_ms and task.status == "completed":
                total_execution_time += task.execution_time_ms
                completed_tasks += 1

        avg_execution_time = (
            total_execution_time / completed_tasks if completed_tasks > 0 else 0
        )

        return {
            "total": len(tasks),
            "by_status": by_status,
            "by_type": by_type,
            "avg_execution_time": avg_execution_time,
        }
