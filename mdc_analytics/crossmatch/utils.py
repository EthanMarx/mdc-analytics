import pandas as pd
import numpy as np
from gwpy.segments import DataQualityDict, DataQualityFlag
from .gracedb import GEVENT_COLUMNS

SEC_PER_DAY = 86164.0905

def apply_skymap_offset(
    events: pd.DataFrame,
    offset: int,
    ra_key: str
) -> pd.DataFrame:
    """
    Corrects injected right ascencsion corresponding to offset,
    adding a `right_ascension_offset` column 
    """    
    skymap_offset = offset % SEC_PER_DAY * 360 / SEC_PER_DAY 
    ra = events[ra_key].values + np.deg2rad(skymap_offset)
    ra = ra % (2 * np.pi)
    events["right_ascension_offset"] = ra
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
            (events[injection_time_key].values[:, None] >= segs[:, 0]) &
            (events[injection_time_key].values[:, None] <= segs[:, 1]),
            axis=1
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
    diffs = np.abs(events.time_geocenter_replay.values[:, None] - pipeline_events.gpstime.values[None, :])
    pipeline_args = np.argmin(np.abs(diffs), axis=0) 
    args = np.argmin(np.abs(diffs), axis=1)
    mins = diffs[np.arange(len(diffs)), args]
    mask = mins < dt 

    # for injections that have a corresponding 
    # gevent, add gevent information, otherwise report `None`
    for attr in GEVENT_COLUMNS.keys():
        output = np.array([None] * len(events))
        output[mask] = pipeline_events.loc[args[mask], attr] 
        events[f"{pipeline}_{attr}"] = output

    events[f"{pipeline}_dt"] = np.abs(events[f"{pipeline}_gpstime"] - events.time_geocenter_replay) 
    
    # calculate mask for pipeline events dataframe that 
    # is true if there was a match with any mdc event
    # within dt threshold
    diffs = diffs.transpose(1, 0)
    pipeline_mins = diffs[np.arange(len(diffs)), pipeline_args]
    found_mask = pipeline_mins < dt
    
    return events, found_mask