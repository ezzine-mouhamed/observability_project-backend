import traceback
from typing import Optional, Any, Dict

from app.utils.logger import get_logger

logger = get_logger(__name__)


class AppException(Exception):
    """
    Base exception for the entire application.

    Auto-logs a structured record at creation time unless `log=False` is passed.
    """

    def __init__(
        self,
        message: str,
        *,
        log: bool = True,
        extra: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(message)
        self.message = message
        self.extra = extra or {}
        self._logged = False

        if log:
            stack = "".join(traceback.format_stack()[:-1])
            log_payload = {
                "exception_type": self.__class__.__name__,
                "exception_message": message,  # Changed from "message" to "exception_message"
                "creation_stack": stack,
                **self.extra,
            }
            logger.error(f"{self.__class__.__name__}: {self.message}", extra=log_payload)
            self._logged = True

    def mark_logged(self) -> None:
        """Mark as logged elsewhere to avoid duplicate logging."""
        self._logged = True