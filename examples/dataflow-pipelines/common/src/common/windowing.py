"""
Windowing utilities for streaming Dataflow pipelines.

This module provides common windowing strategies, triggers, and
time-based processing patterns for real-time data processing.
"""

import apache_beam as beam
from apache_beam import window
from apache_beam.transforms.trigger import (
    AfterWatermark, AfterProcessingTime, AfterCount,
    Repeatedly, AfterAny, AfterAll
)
from apache_beam.transforms import combiners
from typing import Optional, Dict, Any, Callable
from datetime import timedelta
import logging


class WindowingStrategies:
    """
    Common windowing strategies for streaming pipelines.
    """
    
    @staticmethod
    def fixed_windows(window_size_seconds: int) -> window.WindowFn:
        """
        Create fixed time windows.
        
        Args:
            window_size_seconds: Window size in seconds
            
        Returns:
            Fixed window function
        """
        return window.FixedWindows(window_size_seconds)
    
    @staticmethod
    def sliding_windows(
        window_size_seconds: int,
        window_period_seconds: int
    ) -> window.WindowFn:
        """
        Create sliding time windows.
        
        Args:
            window_size_seconds: Window size in seconds
            window_period_seconds: Window slide period in seconds
            
        Returns:
            Sliding window function
        """
        return window.SlidingWindows(window_size_seconds, window_period_seconds)
    
    @staticmethod
    def session_windows(gap_size_seconds: int) -> window.WindowFn:
        """
        Create session windows based on activity gaps.
        
        Args:
            gap_size_seconds: Inactivity gap size in seconds
            
        Returns:
            Session window function
        """
        return window.Sessions(gap_size_seconds)
    
    @staticmethod
    def global_window() -> window.WindowFn:
        """
        Create a global window (no time-based partitioning).
        
        Returns:
            Global window function
        """
        return window.GlobalWindows()


class TriggerStrategies:
    """
    Common trigger strategies for windowing.
    """
    
    @staticmethod
    def watermark_trigger():
        """
        Trigger when watermark passes end of window.
        
        Returns:
            Watermark-based trigger
        """
        return AfterWatermark()
    
    @staticmethod
    def processing_time_trigger(interval_seconds: int):
        """
        Trigger based on processing time intervals.
        
        Args:
            interval_seconds: Trigger interval in seconds
            
        Returns:
            Processing time trigger
        """
        return Repeatedly(AfterProcessingTime(interval_seconds))
    
    @staticmethod
    def count_trigger(element_count: int):
        """
        Trigger after specified number of elements.
        
        Args:
            element_count: Number of elements to trigger
            
        Returns:
            Count-based trigger
        """
        return Repeatedly(AfterCount(element_count))
    
    @staticmethod
    def hybrid_trigger(
        watermark_delay_seconds: Optional[int] = None,
        processing_time_interval_seconds: Optional[int] = None,
        element_count: Optional[int] = None
    ):
        """
        Create hybrid trigger with multiple conditions.
        
        Args:
            watermark_delay_seconds: Watermark delay for late data
            processing_time_interval_seconds: Processing time interval
            element_count: Element count threshold
            
        Returns:
            Hybrid trigger
        """
        triggers = []
        
        # Watermark trigger (possibly with delay)
        if watermark_delay_seconds:
            triggers.append(AfterWatermark(
                early=AfterProcessingTime(processing_time_interval_seconds or 60),
                late=AfterProcessingTime(watermark_delay_seconds)
            ))
        else:
            triggers.append(AfterWatermark())
        
        # Processing time trigger
        if processing_time_interval_seconds:
            triggers.append(AfterProcessingTime(processing_time_interval_seconds))
        
        # Count trigger
        if element_count:
            triggers.append(AfterCount(element_count))
        
        if len(triggers) == 1:
            return triggers[0]
        else:
            return Repeatedly(AfterAny(*triggers))
    
    @staticmethod
    def early_and_late_trigger(
        early_interval_seconds: int = 60,
        late_interval_seconds: int = 300
    ):
        """
        Create early and late firing trigger.
        
        Args:
            early_interval_seconds: Early firing interval
            late_interval_seconds: Late firing interval
            
        Returns:
            Early and late trigger
        """
        return AfterWatermark(
            early=Repeatedly(AfterProcessingTime(early_interval_seconds)),
            late=Repeatedly(AfterProcessingTime(late_interval_seconds))
        )


def create_fixed_windows(
    window_size_seconds: int,
    trigger_interval_seconds: Optional[int] = None,
    allowed_lateness_seconds: int = 3600,
    accumulation_mode: str = "discarding"
) -> beam.PTransform:
    """
    Create a fixed windowing transform with common configurations.
    
    Args:
        window_size_seconds: Window size in seconds
        trigger_interval_seconds: Optional trigger interval
        allowed_lateness_seconds: Allowed lateness for late data
        accumulation_mode: 'accumulating' or 'discarding'
        
    Returns:
        Configured windowing transform
    """
    windowing = beam.WindowInto(
        WindowingStrategies.fixed_windows(window_size_seconds),
        allowed_lateness=timedelta(seconds=allowed_lateness_seconds)
    )
    
    # Add trigger if specified
    if trigger_interval_seconds:
        trigger = TriggerStrategies.hybrid_trigger(
            processing_time_interval_seconds=trigger_interval_seconds
        )
        windowing = windowing.with_trigger(trigger)
    
    # Set accumulation mode
    if accumulation_mode == "accumulating":
        windowing = windowing.accumulating_fired_panes()
    else:
        windowing = windowing.discarding_fired_panes()
    
    return windowing


def create_sliding_windows(
    window_size_seconds: int,
    window_period_seconds: int,
    trigger_interval_seconds: Optional[int] = None,
    allowed_lateness_seconds: int = 3600
) -> beam.PTransform:
    """
    Create a sliding windowing transform.
    
    Args:
        window_size_seconds: Window size in seconds
        window_period_seconds: Window slide period in seconds  
        trigger_interval_seconds: Optional trigger interval
        allowed_lateness_seconds: Allowed lateness for late data
        
    Returns:
        Configured sliding windowing transform
    """
    windowing = beam.WindowInto(
        WindowingStrategies.sliding_windows(window_size_seconds, window_period_seconds),
        allowed_lateness=timedelta(seconds=allowed_lateness_seconds)
    )
    
    if trigger_interval_seconds:
        trigger = TriggerStrategies.processing_time_trigger(trigger_interval_seconds)
        windowing = windowing.with_trigger(trigger)
        windowing = windowing.discarding_fired_panes()
    
    return windowing


def create_session_windows(
    gap_size_seconds: int,
    min_session_length_seconds: Optional[int] = None,
    allowed_lateness_seconds: int = 3600
) -> beam.PTransform:
    """
    Create a session windowing transform.
    
    Args:
        gap_size_seconds: Session gap size in seconds
        min_session_length_seconds: Minimum session length
        allowed_lateness_seconds: Allowed lateness for late data
        
    Returns:
        Configured session windowing transform
    """
    windowing = beam.WindowInto(
        WindowingStrategies.session_windows(gap_size_seconds),
        allowed_lateness=timedelta(seconds=allowed_lateness_seconds)
    )
    
    # Session windows typically use watermark triggers
    trigger = TriggerStrategies.watermark_trigger()
    windowing = windowing.with_trigger(trigger)
    windowing = windowing.accumulating_fired_panes()
    
    return windowing


class WindowedAggregator(beam.DoFn):
    """
    Custom windowed aggregation with metadata.
    """
    
    def __init__(
        self,
        aggregation_functions: Dict[str, Callable],
        window_metadata: bool = True
    ):
        """
        Initialize windowed aggregator.
        
        Args:
            aggregation_functions: Dict of field -> aggregation function
            window_metadata: Whether to include window metadata
        """
        self.aggregation_functions = aggregation_functions
        self.window_metadata = window_metadata
    
    def process(
        self,
        element: Dict[str, Any],
        window=beam.DoFn.WindowParam
    ) -> Dict[str, Any]:
        """
        Process elements with window context.
        
        Args:
            element: Input element
            window: Window parameter
            
        Returns:
            Aggregated result with window metadata
        """
        result = {}
        
        # Apply aggregation functions
        for field, agg_func in self.aggregation_functions.items():
            if field in element:
                try:
                    result[f"{field}_{agg_func.__name__}"] = agg_func(element[field])
                except Exception as e:
                    logging.warning(f"Aggregation failed for {field}: {e}")
                    result[f"{field}_error"] = str(e)
        
        # Add window metadata if requested
        if self.window_metadata:
            result["_window_start"] = window.start.to_utc_datetime().isoformat()
            result["_window_end"] = window.end.to_utc_datetime().isoformat()
            result["_window_type"] = type(window).__name__
        
        return result


class WindowSizeCalculator(beam.DoFn):
    """
    Calculate window sizes and element counts.
    """
    
    def process(
        self,
        elements,
        window=beam.DoFn.WindowParam
    ) -> Dict[str, Any]:
        """
        Calculate window statistics.
        
        Args:
            elements: Collection of elements in window
            window: Window parameter
            
        Returns:
            Window statistics
        """
        element_list = list(elements)
        
        return {
            "window_start": window.start.to_utc_datetime().isoformat(),
            "window_end": window.end.to_utc_datetime().isoformat(),
            "element_count": len(element_list),
            "window_duration_seconds": (window.end - window.start).total_seconds(),
            "elements_per_second": len(element_list) / max(1, (window.end - window.start).total_seconds()),
        }


def create_windowed_aggregation(
    key_field: str,
    value_fields: Dict[str, str],
    window_config: Dict[str, Any]
) -> beam.PTransform:
    """
    Create a windowed aggregation pipeline.
    
    Args:
        key_field: Field to group by
        value_fields: Dict of field -> aggregation type ('sum', 'count', 'avg', etc.)
        window_config: Window configuration
        
    Returns:
        Windowed aggregation transform
    """
    
    def expand(pcoll):
        # Apply windowing
        windowed = pcoll | "Apply Windows" >> create_fixed_windows(**window_config)
        
        # Group by key
        grouped = windowed | f"Group By {key_field}" >> beam.GroupBy(key_field)
        
        # Apply aggregations
        result = grouped
        for field, agg_type in value_fields.items():
            if agg_type == "sum":
                result = result | f"Sum {field}" >> beam.combiners.Sum(field)
            elif agg_type == "count":
                result = result | f"Count {field}" >> beam.combiners.Count()
            elif agg_type == "mean" or agg_type == "avg":
                result = result | f"Mean {field}" >> beam.combiners.Mean(field)
            # Add more aggregation types as needed
        
        return result
    
    return beam.ptransform_fn(expand)()


class LateDataHandler(beam.DoFn):
    """
    Handle late-arriving data in windowed processing.
    """
    
    def __init__(self, late_data_strategy: str = "log"):
        """
        Initialize late data handler.
        
        Args:
            late_data_strategy: Strategy for handling late data
                                ('log', 'drop', 'separate_output')
        """
        self.late_data_strategy = late_data_strategy
    
    def process(
        self,
        element: Dict[str, Any],
        window=beam.DoFn.WindowParam,
        pane_info=beam.DoFn.PaneInfoParam
    ) -> Dict[str, Any]:
        """
        Process elements with late data handling.
        
        Args:
            element: Input element
            window: Window parameter
            pane_info: Pane information
            
        Yields:
            Processed elements
        """
        # Check if this is late data
        is_late = pane_info.is_late if pane_info else False
        
        if is_late:
            if self.late_data_strategy == "log":
                logging.info(f"Processing late data: {element}")
            elif self.late_data_strategy == "drop":
                logging.info(f"Dropping late data: {element}")
                return
            elif self.late_data_strategy == "separate_output":
                element["_is_late_data"] = True
                element["_late_data_timestamp"] = element.get("_pubsub_received_time")
        
        # Add pane information
        if pane_info:
            element["_pane_index"] = pane_info.index
            element["_pane_is_first"] = pane_info.is_first
            element["_pane_is_last"] = pane_info.is_last
            element["_pane_timing"] = str(pane_info.timing)
        
        yield element