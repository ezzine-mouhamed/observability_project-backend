from datetime import datetime
from typing import Any, Dict, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from app.exceptions.validation import ValidationError


class TaskCreate(BaseModel):
    """Schema for task creation request."""

    task_type: str = Field(..., description="Type of task to execute")
    input_data: Optional[Dict[str, Any]] = Field(
        None, description="Input data for the task"
    )
    parameters: Optional[Dict[str, Any]] = Field(
        default_factory=dict, description="Execution parameters"
    )

    @field_validator("task_type")
    @classmethod
    def validate_task_type(cls, v: str) -> str:
        allowed_types = ["summarize", "analyze", "classify", "extract", "translate"]
        if v not in allowed_types:
            raise ValidationError(
                message=f'Task type must be one of: {", ".join(allowed_types)}',
                field="task_type",
                value=v,
                validation_rules={"allowed_types": allowed_types},
                log=False,  # Don't log at validator level
            )
        return v

    @field_validator("parameters")
    @classmethod
    def validate_parameters(
        cls, v: Optional[Dict[str, Any]]
    ) -> Optional[Dict[str, Any]]:
        if v and "max_length" in v:
            if not isinstance(v["max_length"], int) or v["max_length"] <= 0:
                raise ValidationError(
                    message="max_length must be a positive integer",
                    field="parameters.max_length",
                    value=v.get("max_length"),
                    validation_rules={"type": "int", "min": 1},
                    log=False,
                )
        return v

    @model_validator(mode="before")
    @classmethod
    def normalize_input_data(cls, data: Any) -> Any:
        """Normalize input data before validation."""
        if isinstance(data, dict):
            # Handle 'payload' field as alias for 'input_data'
            if "payload" in data and "input_data" not in data:
                data["input_data"] = data.pop("payload")
            # Ensure parameters exists
            if "parameters" not in data:
                data["parameters"] = {}
        return data


class TaskResponse(BaseModel):
    """Schema for task response."""

    id: int
    task_type: str
    status: str
    input_data: Optional[Dict[str, Any]]
    parameters: Dict[str, Any]
    output_data: Optional[Dict[str, Any]]
    error_message: Optional[str]
    execution_time_ms: Optional[int]
    created_at: datetime
    started_at: Optional[datetime]
    completed_at: Optional[datetime]
    quality_score: Optional[float]

    model_config = ConfigDict(from_attributes=True)


class TraceInfo(BaseModel):
    """Schema for trace information."""

    trace_id: str
    operation: str
    start_time: datetime
    end_time: Optional[datetime]
    duration_ms: Optional[int]
    success: bool
    decision_count: int
    event_count: int
