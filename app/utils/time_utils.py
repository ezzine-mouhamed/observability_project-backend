import functools
import time
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, Optional

from app.utils.logger import get_logger

logger = get_logger(__name__)


class Timer:
    def __init__(self, name: str = "operation"):
        self.name = name
        self.start_time: Optional[float] = None
        self.end_time: Optional[float] = None

    def __enter__(self):
        self.start_time = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end_time = time.perf_counter()
        duration_ms = self.elapsed_ms()

        logger.debug(
            f"Timer completed: {self.name}",
            extra={
                "timer_name": self.name,
                "duration_ms": duration_ms,
                "start_time": self.start_time,
                "end_time": self.end_time,
            },
        )

    def elapsed_ms(self) -> float:
        if self.start_time is None:
            return 0.0
        end = self.end_time or time.perf_counter()
        return (end - self.start_time) * 1000

    def elapsed_seconds(self) -> float:
        return self.elapsed_ms() / 1000


def timed(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        timer = Timer(func.__name__)
        with timer:
            result = func(*args, **kwargs)
        return result

    return wrapper


def format_duration_ms(duration_ms: float) -> str:
    if duration_ms < 1000:
        return f"{duration_ms:.0f}ms"
    elif duration_ms < 60000:
        return f"{duration_ms/1000:.1f}s"
    else:
        minutes = duration_ms / 60000
        return f"{minutes:.1f}min"


def get_current_utc() -> datetime:
    return datetime.now(timezone.utc)


def get_utc_from_timestamp(timestamp: float) -> datetime:
    return datetime.fromtimestamp(timestamp, timezone.utc)


def format_datetime_iso(dt: datetime) -> str:
    return dt.isoformat()


def parse_iso_datetime(iso_string: str) -> datetime:
    if iso_string.endswith("Z"):
        iso_string = iso_string[:-1]
    return datetime.fromisoformat(iso_string).replace(tzinfo=timezone.utc)


def is_within_time_range(
    check_time: datetime, start_time: datetime, end_time: datetime
) -> bool:
    return start_time <= check_time <= end_time


def time_window_from_now(hours: int = 1) -> tuple[datetime, datetime]:
    end_time = get_current_utc()
    start_time = end_time - timedelta(hours=hours)
    return start_time, end_time


@contextmanager
def execution_timer(operation_name: str, context: Optional[Dict[str, Any]] = None):
    timer = Timer(operation_name)
    start_time = get_current_utc()

    try:
        with timer:
            yield timer
    finally:
        duration_ms = timer.elapsed_ms()

        log_data = {
            "operation": operation_name,
            "duration_ms": duration_ms,
            "start_time": format_datetime_iso(start_time),
            "end_time": format_datetime_iso(get_current_utc()),
            "formatted_duration": format_duration_ms(duration_ms),
        }

        if context:
            log_data["context"] = context

        logger.info(f"Execution timer: {operation_name}", extra=log_data)
