from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, ConfigDict 


class DecisionRecord(BaseModel):
    """Schema for decision record."""

    type: str
    timestamp: datetime
    context: Dict[str, Any]


class EventRecord(BaseModel):
    """Schema for event record."""

    type: str
    timestamp: datetime
    data: Dict[str, Any]


class ErrorRecord(BaseModel):
    """Schema for error record."""

    type: str
    message: str
    timestamp: datetime
    context: Optional[Dict[str, Any]]


class TraceDetail(BaseModel):
    """Schema for detailed trace response."""

    id: int
    trace_id: str
    parent_trace_id: Optional[str]
    task_id: Optional[int]
    operation: str
    context: Dict[str, Any]
    decisions: List[DecisionRecord] = Field(default_factory=list)
    events: List[EventRecord] = Field(default_factory=list)
    error: Optional[ErrorRecord]
    start_time: datetime
    end_time: Optional[datetime]
    duration_ms: Optional[int]
    success: bool
    created_at: datetime

    model_config = ConfigDict(from_attributes=True)


class TraceSummary(BaseModel):
    """Schema for trace summary."""

    trace_id: str
    operation: str
    success: bool
    duration_ms: Optional[int]
    decision_count: int
    event_count: int
    has_error: bool
    start_time: datetime
