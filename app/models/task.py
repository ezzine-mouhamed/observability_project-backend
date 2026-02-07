from datetime import datetime, timezone
from typing import Any, Dict, Optional

from sqlalchemy import JSON, Integer, String, Text, DateTime, Float
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.extensions import db


class Task(db.Model):
    __tablename__ = "tasks"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    task_type: Mapped[str] = mapped_column(String(100), nullable=False)
    input_data: Mapped[Optional[Dict[str, Any]]] = mapped_column(JSON, nullable=True)
    parameters: Mapped[Dict[str, Any]] = mapped_column(JSON, default=dict)
    output_data: Mapped[Optional[Dict[str, Any]]] = mapped_column(JSON, nullable=True)
    error_message: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    status: Mapped[str] = mapped_column(String(50), default="pending")
    execution_time_ms: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(timezone.utc)
    )
    started_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    completed_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    # NEW: Quality and performance fields
    quality_score: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    complexity_level: Mapped[Optional[str]] = mapped_column(String(50), nullable=True)
    agent_involved: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)
    validation_results: Mapped[Optional[Dict[str, Any]]] = mapped_column(JSON, default=dict)
    performance_metrics: Mapped[Optional[Dict[str, Any]]] = mapped_column(JSON, default=dict)

    traces = relationship(
        "ExecutionTrace",
        back_populates="task",
        lazy="select",
        cascade="all, delete-orphan",
    )

    def __repr__(self) -> str:
        return f"<Task {self.id} ({self.task_type}) - {self.status} quality:{self.quality_score or 'N/A'}>"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "task_type": self.task_type,
            "input_data": self.input_data,
            "parameters": self.parameters,
            "output_data": self.output_data,
            "error_message": self.error_message,
            "status": self.status,
            "execution_time_ms": self.execution_time_ms,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": (
                self.completed_at.isoformat() if self.completed_at else None
            ),
            "quality_score": self.quality_score,
            "complexity_level": self.complexity_level,
            "agent_involved": self.agent_involved,
            "validation_results": self.validation_results or {},
            "performance_metrics": self.performance_metrics or {},
            "trace_count": len(self.traces),
        }

    def start_execution(self) -> None:
        self.started_at = datetime.now(timezone.utc)
        self.status = "running"

    def complete_execution(
        self, 
        success: bool, 
        output: Optional[Any] = None, 
        error: Optional[str] = None
    ) -> None:
        self.completed_at = datetime.now(timezone.utc)
        self.status = "completed" if success else "failed"
        self.output_data = output
        self.error_message = error

        if self.started_at and self.completed_at:
            # Ensure both are timezone-aware for subtraction
            started_at = self.started_at
            completed_at = self.completed_at
            
            if started_at.tzinfo is None:
                started_at = started_at.replace(tzinfo=timezone.utc)
            
            if completed_at.tzinfo is None:
                completed_at = completed_at.replace(tzinfo=timezone.utc)
            
            self.execution_time_ms = int(
                (completed_at - started_at).total_seconds() * 1000
            )
    
    def calculate_quality_score(self) -> Optional[float]:
        """
        Calculate quality score based on traces and validation results.
        
        Returns:
            Quality score between 0.0 and 1.0
        """
        if not self.traces:
            return None
        
        # Calculate from trace quality metrics
        trace_quality_scores = []
        for trace in self.traces:
            if trace.quality_metrics and "quality_score" in trace.quality_metrics:
                trace_quality_scores.append(trace.quality_metrics["quality_score"])
        
        if trace_quality_scores:
            # Also consider validation results
            validation_score = 1.0
            if self.validation_results:
                # Example: check if all validations passed
                passed = self.validation_results.get("passed", 0)
                total = self.validation_results.get("total", 0)
                validation_score = passed / total if total > 0 else 1.0
            
            # Weighted average
            trace_avg = sum(trace_quality_scores) / len(trace_quality_scores)
            self.quality_score = (trace_avg * 0.7) + (validation_score * 0.3)
        
        return self.quality_score
    
    def record_validation(self, check_name: str, passed: bool, details: Optional[Dict] = None) -> None:
        """Record a validation result."""
        if not self.validation_results:
            self.validation_results = {
                "checks": [],
                "passed": 0,
                "total": 0,
                "score": 0.0,
            }
        
        validation_record = {
            "check_name": check_name,
            "passed": passed,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "details": details or {},
        }
        
        self.validation_results["checks"].append(validation_record)
        self.validation_results["total"] += 1
        if passed:
            self.validation_results["passed"] += 1
        
        # Update score
        self.validation_results["score"] = (
            self.validation_results["passed"] / self.validation_results["total"]
            if self.validation_results["total"] > 0 else 0.0
        )
    
    def record_performance_metric(self, metric_name: str, value: Any) -> None:
        """Record a performance metric."""
        if not self.performance_metrics:
            self.performance_metrics = {}
        
        self.performance_metrics[metric_name] = {
            "value": value,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
