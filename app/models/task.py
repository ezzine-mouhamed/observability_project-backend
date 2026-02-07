from datetime import datetime, timezone
from typing import Any, Dict, Optional

from sqlalchemy import JSON, Integer, String, Text, DateTime
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

    traces = relationship(
        "ExecutionTrace",
        back_populates="task",
        lazy="select",
        cascade="all, delete-orphan",
    )

    def __repr__(self) -> str:
        return f"<Task {self.id} ({self.task_type}) - {self.status}>"

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
            
            # If started_at is naive, make it aware
            if started_at.tzinfo is None:
                started_at = started_at.replace(tzinfo=timezone.utc)
            
            # If completed_at is naive, make it aware  
            if completed_at.tzinfo is None:
                completed_at = completed_at.replace(tzinfo=timezone.utc)
            
            self.execution_time_ms = int(
                (completed_at - started_at).total_seconds() * 1000
            )
