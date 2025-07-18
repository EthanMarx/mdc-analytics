import pandas as pd
import numpy as np
import logging
from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp
from gwpy.segments import DataQualityFlag
from .gracedb import GEVENT_COLUMNS

SEC_PER_DAY = 86164.0905

# Global process pool for parallel processing
_global_pool = None
mp.set_start_method("spawn", force=True)


def get_pool(max_workers=None):
    """Get or create a global process pool."""
    global _global_pool
    if _global_pool is None:
        logging.info("Initializing pool")
        _global_pool = ProcessPoolExecutor(max_workers=max_workers)
    return _global_pool


def shutdown_global_pool():
    """Shutdown the global process pool."""
    global _global_pool
    if _global_pool is not None:
        _global_pool.shutdown()
        _global_pool = None


def parallelize(
    func: callable,
    events: pd.DataFrame,
    max_workers=15,
):
    """Submit function calls to the global process pool for parallel execution."""  # noqa: E501
    executor = get_pool(max_workers)
    future_to_index = {
        executor.submit(func, row): i
        for i, (_, row) in enumerate(events.iterrows())
    }
    return future_to_index


def apply_skymap_offset(
    events: pd.DataFrame,
    offset: int,
    ra_key: str,
    column_name: str = "right_ascension_offset",
) -> pd.DataFrame:
    """
    Corrects injected right ascension corresponding to offset,
    adding a column with the specified name
    """
    skymap_offset = offset % SEC_PER_DAY * 360 / SEC_PER_DAY
    ra = events[ra_key].values + np.deg2rad(skymap_offset)
    ra = ra % (2 * np.pi)
    events[column_name] = ra
    return events


def append_data_quality_flags(
    events: pd.DataFrame,
    flags: list[str],
    start: float,
    stop: float,
    injection_time_key: str,
) -> tuple[pd.DataFrame, float]:
    """
    For each flag, adds a boolean column to the events dataframe indicating if
    an injection occured during that flags active segments
    """
    mask = np.ones(len(events), dtype=bool)

    for flag in flags:
        dq_flag = DataQualityFlag.query(flag, start, stop)
        segs = np.asarray(dq_flag.active)
        mask = np.any(
            (events[injection_time_key].values[:, None] >= segs[:, 0])
            & (events[injection_time_key].values[:, None] <= segs[:, 1]),
            axis=1,
        )

        events[flag] = mask

    return events


def crossmatch_gevents(
    events: pd.DataFrame,
    pipeline_events: pd.DataFrame,
    pipeline: str,
    dt: float,
) -> tuple[pd.DataFrame, pd.Series]:
    # calculate mask for ground truth events dataframe that
    # is true if there was a match with any gevent event
    # within dt threshold
    diffs = np.abs(
        events[f"{pipeline}_time_geocenter_replay"].values[:, None]
        - pipeline_events.gpstime.values[None, :]
    )
    pipeline_args = np.argmin(np.abs(diffs), axis=0)
    args = np.argmin(np.abs(diffs), axis=1)
    mins = diffs[np.arange(len(diffs)), args]
    mask = mins < dt

    # Collect all new columns to add at once to avoid fragmentation
    new_columns = {}

    for attr in GEVENT_COLUMNS.keys():
        # Extract dtype and default value from GEVENT_COLUMNS tuple
        dtype, default_val = GEVENT_COLUMNS[attr]

        output = np.full(len(events), default_val, dtype=object)
        output[mask] = pipeline_events.loc[args[mask], attr]
        new_columns[f"{pipeline}_{attr}"] = pd.Series(output).astype(dtype)

    # Handle preferred_pipeline column for preferred events
    if (
        pipeline == "preferred"
        and "preferred_pipeline" in pipeline_events.columns
    ):
        output = np.full(len(events), "", dtype=object)
        output[mask] = pipeline_events.loc[args[mask], "preferred_pipeline"]
        new_columns["preferred_pipeline"] = pd.Series(output).astype("string")

    # Calculate dt column
    new_columns[f"{pipeline}_dt"] = np.abs(
        new_columns[f"{pipeline}_gpstime"]
        - events[f"{pipeline}_time_geocenter_replay"]
    )
    # Add all new columns at once using concat to avoid fragmentation
    new_df = pd.DataFrame(new_columns, index=events.index)
    events = pd.concat([events, new_df], axis=1)

    # calculate mask for pipeline events dataframe that
    # is true if there was a match with any mdc event
    # within dt threshold
    diffs = diffs.transpose(1, 0)
    pipeline_mins = diffs[np.arange(len(diffs)), pipeline_args]
    found_mask = pipeline_mins < dt

    return events, found_mask


def filter_events(
    events: pd.DataFrame, filters: list[tuple[str, float, float]] = None
) -> pd.DataFrame:
    """
    Apply filters to the events DataFrame.
    Filters should be a list of tuples with (column, min, max).

    Args:
        events: DataFrame to filter
        filters: List of tuples (column_name, min_value, max_value)

    Returns:
        Filtered DataFrame
    """
    if filters is None:
        return events

    for key, low, high in filters:
        logging.info(f"Applying filter on {key} to range ({low}, {high})")
        low, high = float(low), float(high)
        mask = (events[key] >= low) & (events[key] <= high)
        events = events[mask]

    events.reset_index(drop=True, inplace=True)
    return events
