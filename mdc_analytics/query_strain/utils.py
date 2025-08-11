from gwpy.timeseries import TimeSeries
import lal
from typing import Literal
import numpy as np
from pathlib import Path
import pandas as pd
from collections import defaultdict
import re
import logging
from gwpy.segments import DataQualityFlag

patterns = {
    "prefix": "[a-zA-Z0-9_:-]+",
    "start": "[0-9]{10}",
    "duration": "[1-9][0-9]*",
    "suffix": "(gwf)|(hdf5)|(h5)",
}
groups = {k: f"(?P<{k}>{v})" for k, v in patterns.items()}
pattern = "{prefix}-{start}-{duration}.{suffix}".format(**groups)
fname_re = re.compile(pattern)


def resample(
    data: TimeSeries,
    sample_rate: float,
    method: Literal["lal", "gwpy"] = "gwpy",
):
    if method == "gwpy":
        data = data.resample(sample_rate)
    elif method == "lal":
        data = data.astype(np.float64).to_lal()
        lal.ResampleREAL8TimeSeries(data, float(1 / sample_rate))
        data = TimeSeries(
            data.data.data,
            epoch=data.epoch,
            dt=data.deltaT,
        )
    return data


def parse_fnames(ifos: list[str], data_dirs: dict[str, Path]):
    """
    Parse filename directory for each ifo into
    list of tuples of (filename, start, end)
    so that we can easily find relevant files later
    """
    output = defaultdict(list)
    for ifo in ifos:
        data_dir = data_dirs[ifo]
        for file in sorted(data_dir.iterdir()):
            match = fname_re.search(file.name)
            _, start, duration, *_ = match.groups()
            start = int(start)
            duration = int(duration)
            output[ifo].append((file, start, duration + start))
    return output


def read_data(
    frame_files: list[tuple[Path, float, float]],
    channel: str,
    start: float,
    end: float,
    nproc: int = 1,
):
    paths = []
    logging.info(f"Reading data from {start} to {end}")
    # find frame files that overlap with the start and end time
    for file, frame_start, frame_end in sorted(frame_files):
        if int(frame_start) <= start and start <= int(frame_end):
            paths.append(str(file))
        elif int(frame_start) <= end and end <= int(frame_end):
            paths.append(str(file))

    if len(paths) == 1:
        data = TimeSeries.read(paths[0], channel=channel, nproc=nproc)
    elif len(paths) == 2:
        data = TimeSeries.read(paths[0], channel=channel, nproc=nproc)
        second = TimeSeries.read(paths[1], channel=channel, nproc=nproc)
        data.append(second)
    else:
        raise ValueError("Shouldn't be more than 2 files")

    data = data.crop(start, end)
    return data


def query_event_strain(
    row: tuple[int, pd.Series],
    ifos: list[str],
    channels: list[str],
    fname_data,
    sample_rate: float,
    psd_length: float,
    resample_method: str,
):
    event_index, event = row
    gpstime = event.time
    start = gpstime - psd_length - 4
    end = gpstime + psd_length + 4

    event_strain = {}
    for channel, ifo in zip(channels, ifos, strict=False):
        logging.debug(f"Reading {ifo} data for event {event_index}")
        data = read_data(fname_data[ifo], channel, start, end)
        data = resample(data, sample_rate, method=resample_method)
        event_strain[ifo] = data

    return event_index, event_strain


def filter_events(
    events: pd.DataFrame, filters: list[tuple[str, float, float]]
):
    """
    Apply filters to the events DataFrame.
    Filters can be a list of strings
    or a list of tuples with (column, min, max).
    """
    if filters is None:
        return events

    for key, low, high in filters:
        logging.info(f"Applying filter on {key} to range ({low}, {high})")
        low, high = float(low), float(high)
        mask = (events[key] >= low) & (events[key] <= high)
        events = events[mask]

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
