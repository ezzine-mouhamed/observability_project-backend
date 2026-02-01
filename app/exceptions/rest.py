from typing import Optional, Any, Dict
from .base import AppException


class RestAPIError(AppException):
    """
    Generic exception for REST/API response errors.
    
    Attributes:
        message: human-readable error message
        status_code: HTTP status code or error code
        response: raw response content (if available)
    """
    
    def __init__(
        self,
        message: str,
        status_code: Optional[int] = None,
        response: Optional[Dict[str, Any] | str] = None,
        *,
        log: bool = True,
        extra: Optional[Dict[str, Any]] = None,
    ):
        self.status_code = status_code
        self.response = response
        
        # Build extra context including status_code and response
        api_extra = extra or {}
        api_extra.update({
            "status_code": status_code,
            "response": response,
        })
        
        super().__init__(message, log=log, extra=api_extra)