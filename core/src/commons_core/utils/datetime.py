"""DateTime utilities for timezone handling and formatting."""

from datetime import datetime, timezone, timedelta
from typing import Optional, Union
import pytz
from dateutil import parser
from zoneinfo import ZoneInfo


def now_utc() -> datetime:
    """Get current UTC datetime."""
    return datetime.now(timezone.utc)


def parse_datetime(
    date_string: str,
    tz: Optional[Union[str, timezone]] = None,
    fuzzy: bool = True,
) -> datetime:
    """Parse datetime string with optional timezone."""
    # Parse the datetime
    dt = parser.parse(date_string, fuzzy=fuzzy)
    
    # Handle timezone
    if tz:
        if isinstance(tz, str):
            tz = ZoneInfo(tz)
        
        if dt.tzinfo is None:
            # Naive datetime, localize it
            dt = dt.replace(tzinfo=tz)
        else:
            # Convert to target timezone
            dt = dt.astimezone(tz)
            
    return dt


def format_datetime(
    dt: datetime,
    format: str = "%Y-%m-%d %H:%M:%S",
    tz: Optional[Union[str, timezone]] = None,
) -> str:
    """Format datetime with optional timezone conversion."""
    if tz:
        if isinstance(tz, str):
            tz = ZoneInfo(tz)
        dt = dt.astimezone(tz)
        
    return dt.strftime(format)


def to_timestamp(dt: datetime) -> float:
    """Convert datetime to Unix timestamp."""
    return dt.timestamp()


def from_timestamp(timestamp: float, tz: Optional[Union[str, timezone]] = None) -> datetime:
    """Convert Unix timestamp to datetime."""
    dt = datetime.fromtimestamp(timestamp, tz=timezone.utc)
    
    if tz:
        if isinstance(tz, str):
            tz = ZoneInfo(tz)
        dt = dt.astimezone(tz)
        
    return dt


def timezone_aware(dt: datetime, tz: Optional[Union[str, timezone]] = None) -> datetime:
    """Ensure datetime is timezone-aware."""
    if dt.tzinfo is None:
        # Naive datetime
        if tz is None:
            tz = timezone.utc
        elif isinstance(tz, str):
            tz = ZoneInfo(tz)
        return dt.replace(tzinfo=tz)
    return dt


def relative_time(dt: datetime, reference: Optional[datetime] = None) -> str:
    """Get human-readable relative time."""
    if reference is None:
        reference = now_utc()
        
    # Ensure both are timezone-aware
    dt = timezone_aware(dt)
    reference = timezone_aware(reference)
    
    delta = reference - dt
    
    if delta.days > 365:
        years = delta.days // 365
        return f"{years} year{'s' if years > 1 else ''} ago"
    elif delta.days > 30:
        months = delta.days // 30
        return f"{months} month{'s' if months > 1 else ''} ago"
    elif delta.days > 0:
        return f"{delta.days} day{'s' if delta.days > 1 else ''} ago"
    elif delta.seconds > 3600:
        hours = delta.seconds // 3600
        return f"{hours} hour{'s' if hours > 1 else ''} ago"
    elif delta.seconds > 60:
        minutes = delta.seconds // 60
        return f"{minutes} minute{'s' if minutes > 1 else ''} ago"
    else:
        return "just now"


def business_days_between(start: datetime, end: datetime) -> int:
    """Calculate business days between two dates."""
    if start > end:
        start, end = end, start
        
    days = 0
    current = start.date()
    end_date = end.date()
    
    while current <= end_date:
        if current.weekday() < 5:  # Monday = 0, Friday = 4
            days += 1
        current += timedelta(days=1)
        
    return days


def next_business_day(dt: datetime) -> datetime:
    """Get the next business day."""
    next_day = dt + timedelta(days=1)
    
    # Skip to Monday if it's weekend
    while next_day.weekday() >= 5:
        next_day += timedelta(days=1)
        
    return next_day.replace(hour=9, minute=0, second=0, microsecond=0)