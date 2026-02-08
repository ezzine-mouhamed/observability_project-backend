from datetime import datetime, timezone
from typing import Any, Dict, Optional

from sqlalchemy import JSON, Float, Integer, String, DateTime, Boolean, ForeignKey
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
    agent_observations: Mapped[Optional[Dict[str, Any]]] = mapped_column(JSON, default=list)
    quality_metrics: Mapped[Optional[Dict[str, Any]]] = mapped_column(JSON, default=dict)
    agent_context: Mapped[Optional[Dict[str, Any]]] = mapped_column(JSON, default=dict)
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
    # Performance and quality indicators
    performance_score: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    complexity_score: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    efficiency_score: Mapped[Optional[float]] = mapped_column(Float, nullable=True)

    task = relationship("Task", back_populates="traces")

    def __repr__(self) -> str:
        quality = self.quality_metrics.get("quality_score", 0) if self.quality_metrics else 0
        return f"<Trace {self.trace_id} ({self.operation}) - success:{self.success} quality:{quality:.2f}>"

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
            "agent_observations": self.agent_observations or [],
            "quality_metrics": self.quality_metrics or {},
            "agent_context": self.agent_context or {},
            "error": self.error,
            "start_time": self.start_time.isoformat() if self.start_time else None,
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "duration_ms": self.duration_ms,
            "success": self.success,
            "created_at": self.created_at.isoformat(),
            "performance_score": self.performance_score,
            "complexity_score": self.complexity_score,
            "efficiency_score": self.efficiency_score,
        }

    def calculate_performance_score(self) -> float:
        """Calculate and set performance score."""
        scores = []
        
        if self.success:
            scores.append(1.0)
        
        if self.duration_ms:
            # Normalize: 0-10 seconds = 1.0-0.0
            duration_score = max(0.0, 1.0 - (self.duration_ms / 10000))
            scores.append(duration_score)
        
        if self.quality_metrics and "quality_score" in self.quality_metrics:
            scores.append(self.quality_metrics["quality_score"])
        
        self.performance_score = sum(scores) / len(scores) if scores else 0.5
        return self.performance_score
