from datetime import datetime, timezone
from typing import Any, Dict, Optional

from sqlalchemy import JSON, String, DateTime, Integer, Text
from sqlalchemy.orm import Mapped, mapped_column

from app.extensions import db


class AgentInsight(db.Model):
    __tablename__ = "agent_insights"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    agent_name: Mapped[str] = mapped_column(String(100), nullable=False, index=True)
    insight_type: Mapped[str] = mapped_column(String(50), nullable=False, index=True)
    insight_data: Mapped[Dict[str, Any]] = mapped_column(JSON, default=dict)
    source_trace_id: Mapped[Optional[str]] = mapped_column(String(36), nullable=True)
    source_task_id: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    confidence_score: Mapped[Optional[float]] = mapped_column(db.Float, nullable=True)
    impact_prediction: Mapped[Optional[str]] = mapped_column(String(20), nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(timezone.utc)
    )
    applied_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    applied_result: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    def __repr__(self) -> str:
        return f"<AgentInsight {self.id} ({self.insight_type}) for {self.agent_name}>"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "agent_name": self.agent_name,
            "insight_type": self.insight_type,
            "insight_data": self.insight_data,
            "source_trace_id": self.source_trace_id,
            "source_task_id": self.source_task_id,
            "confidence_score": self.confidence_score,
            "impact_prediction": self.impact_prediction,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "applied_at": self.applied_at.isoformat() if self.applied_at else None,
            "applied_result": self.applied_result,
            "is_applied": self.applied_at is not None,
        }
    
    def mark_applied(self, result: str = "success") -> None:
        self.applied_at = datetime.now(timezone.utc)
        self.applied_result = result
    
    def calculate_impact_prediction(self) -> str:
        if self.insight_type in ["quality_improvement", "efficiency_gain"]:
            return "high"
        elif self.insight_type in ["behavior_pattern", "optimization_suggestion"]:
            return "medium"
        else:
            return "low"