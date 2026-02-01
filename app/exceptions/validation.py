from typing import Optional, Any, Dict
from .base import AppException


class ValidationError(AppException):
    """Exception for validation errors."""
    
    def __init__(
        self,
        message: str,
        *,
        field: Optional[str] = None,
        value: Optional[Any] = None,
        validation_rules: Optional[Dict[str, Any]] = None,
        log: bool = True,
        extra: Optional[Dict[str, Any]] = None,
    ):
        self.field = field
        self.value = value
        self.validation_rules = validation_rules or {}
        
        # Build extra context
        validation_extra = extra or {}
        validation_extra.update({
            "field": field,
            "value": str(value) if value is not None else None,
            "validation_rules": validation_rules,
        })
        
        super().__init__(message, log=log, extra=validation_extra)