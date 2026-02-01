from datetime import datetime, timezone
from typing import Any, Dict, Optional

from sqlalchemy import JSON, Integer, String, DateTime, Boolean, ForeignKey
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.extensions import db


class ExecutionTrace(db.Model):
    __tablename__ = "execution_traces"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    trace_id: Mapped[str] = mapped_column(String(36), unique=True, nullable=False)
    parent_trace_id: Mapped[Optional[str]] = mapped_column(String(36), nullable=True)
    task_id: Mapped[Optional[int]] = mapped_column(
        Integer, ForeignKey("tasks.id"), nullable=True
    )
    operation: Mapped[str] = mapped_column(String(200), nullable=False)
    context: Mapped[Dict[str, Any]] = mapped_column(JSON, default=dict)
    decisions: Mapped[Optional[Dict[str, Any]]] = mapped_column(JSON, default=list)
    events: Mapped[Optional[Dict[str, Any]]] = mapped_column(JSON, default=list)
    error: Mapped[Optional[Dict[str, Any]]] = mapped_column(JSON, nullable=True)
    start_time: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    end_time: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    duration_ms: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    success: Mapped[bool] = mapped_column(Boolean, default=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(timezone.utc)
    )

    task = relationship("Task", back_populates="traces")

    def __repr__(self) -> str:
        return f"<Trace {self.trace_id} ({self.operation}) - {'success' if self.success else 'failed'}>"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "trace_id": self.trace_id,
            "parent_trace_id": self.parent_trace_id,
            "task_id": self.task_id,
            "operation": self.operation,
            "context": self.context,
            "decisions": self.decisions or [],
            "events": self.events or [],
            "error": self.error,
            "start_time": self.start_time.isoformat() if self.start_time else None,
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "duration_ms": self.duration_ms,
            "success": self.success,
            "created_at": self.created_at.isoformat(),
        }
