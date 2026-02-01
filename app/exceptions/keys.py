from enum import Enum, unique

@unique
class RestErrorKey(str, Enum):
    """Centralized keys for REST/API errors."""

    MODEL_NOT_FOUND = "MODEL_NOT_FOUND"
    BAD_REQUEST = "BAD_REQUEST"
    TIMEOUT = "TIMEOUT"
    CONNECTION_FAILED = "CONNECTION_FAILED"
    API_ERROR = "API_ERROR"
