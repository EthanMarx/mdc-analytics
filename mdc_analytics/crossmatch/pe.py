import logging
import traceback
from io import BytesIO
from functools import partial
from concurrent.futures import as_completed

import h5py
import numpy as np
import pandas as pd
import requests
from ligo.gracedb.rest import GraceDb
from ligo.skymap.io import events
from igwn_ligolw.utils import load_fileobj
from igwn_ligolw import lsctables
from tqdm.auto import tqdm

from .utils import parallelize

PE_KEYS = ["chirp_mass", "luminosity_distance"]
MATCHED_FILTERING_PIPELINES = ["mbta", "pycbc", "gstlal", "spiir"]


def _aframe_wrapper(row, gdb_server):
    """Wrapper function for aframe PE processing."""
    gid = row.aframe_graceid
    return _process_posterior(gid, gdb_server)


def _cwb_wrapper(row, gdb_server):
    """Wrapper function for cwb PE processing."""
    gid = row.cwb_graceid
    return _process_cwb(gid, gdb_server)


def _matched_filter_wrapper(row, pipeline, gdb_server):
    """Wrapper function for matched filter PE processing."""
    gid = getattr(row, f"{pipeline}_graceid")
    return _process_coinc(gid, gdb_server)


def _process_posterior(
    gid: str,
    gdb_server: str,
):
    if not gid:
        return None

    gdb = GraceDb(service_url=gdb_server, use_auth="scitoken")
    try:
        response = gdb.files(gid, "amplfi.posterior_samples.hdf5")
    except requests.exceptions.HTTPError as e:
        if e.response.status_code == 404:
            logging.warning(
                f"File 'amplfi.posterior_samples.hdf5' not found for event {gid}"
            )
            return None
        else:
            raise
    else:
        posterior = BytesIO(response.read())
        
        output = {}
        with h5py.File(posterior) as f:
            for key in PE_KEYS:
                samples = f["posterior_samples"][key][:]
                output[key] = np.median(samples)
                output[f"{key}_lower"] = np.percentile(samples, 5)
                output[f"{key}_upper"] = np.percentile(samples, 95)
        return output

def process_aframe_pe(
    events: pd.DataFrame,
    gdb_server: str,
    max_workers: int = 15,
    raise_on_error: bool = False,
):
    func = partial(_aframe_wrapper, gdb_server=gdb_server)
    futures = parallelize(func, events, max_workers=max_workers)
    results = [None] * len(events)
    for future in tqdm(
        as_completed(futures),
        total=len(futures),
        desc="Processing Aframe PE",
    ):
        idx = futures[future]
        try:
            results[idx] = future.result()
        except Exception as e:
            if raise_on_error:
                logging.error(
                    f"Failed to query posterior for "
                    f"{events.aframe_graceid.values[idx]}"
                )
                logging.error(traceback.format_exc())
                raise e
            else:
                logging.error(
                    f"Failed to query posterior for "
                    f"{events.aframe_graceid.values[idx]}"
                )
                results[idx] = None
    valid_results = [r for r in results if r is not None]
    all_keys = valid_results[0].keys() if valid_results else []

    for key in all_keys:
        output = []
        for result in results:
            output.append(result[key] if result is not None else np.nan)
        events["aframe_" + key] = output

    return events


def _process_coinc(
    gid: str,
    gdb_server: str,
):
    if not gid:
        return None

    gdb = GraceDb(service_url=gdb_server, use_auth="scitoken")
    try:
        response = gdb.files(gid, "coinc.xml")
    except requests.exceptions.HTTPError as e:
        if e.response.status_code == 404:
            logging.warning(f"File 'coinc.xml' not found for event {gid}")
            return None
        else:
            raise
    else:
        coinc_data = BytesIO(response.read())
        doc = load_fileobj(
            coinc_data, contenthandler=events.ligolw.ContentHandler
        )
        inspiral_table = lsctables.SnglInspiralTable.get_table(doc)

        output = {}
        for coinc_row in inspiral_table:
            output["chirp_mass"] = coinc_row.mchirp
            output["mass1"] = coinc_row.mass1
            output["mass2"] = coinc_row.mass2
        return output


def process_matched_filter_pe(
    events: pd.DataFrame,
    gdb_server: str,
    pipeline: str,
    max_workers: int = 15,
    raise_on_error: bool = False,
):
    func = partial(
        _matched_filter_wrapper, pipeline=pipeline, gdb_server=gdb_server
    )
    futures = parallelize(func, events, max_workers=max_workers)
    results = [None] * len(events)
    gids = getattr(events, f"{pipeline}_graceid").values
    for future in tqdm(
        as_completed(futures),
        total=len(futures),
        desc=f"Processing {pipeline} PE",
    ):
        idx = futures[future]
        try:
            results[idx] = future.result()
        except Exception as e:
            if raise_on_error:
                logging.error(f"Failed to parse coinc for {gids[idx]}")
                logging.error(traceback.format_exc())
                raise e
            else:
                logging.error(f"Failed to parse coinc for {gids[idx]}")
                logging.error(traceback.format_exc())
                results[idx] = None
    for key in ["chirp_mass", "mass1", "mass2"]:
        output = []
        for result in results:
            output.append(result[key] if result is not None else np.nan)
        events[f"{pipeline}_" + key] = output

    return events


def _process_cwb(
    gid: str,
    gdb_server: str,
):
    if not gid:
        return None

    gdb = GraceDb(service_url=gdb_server, use_auth="scitoken")
    try:
        response = gdb.files(gid, "trigger.txt")
    except requests.exceptions.HTTPError as e:
        if e.response.status_code == 404:
            logging.warning(f"File 'trigger.txt' not found for event {gid}")
            return None
        else:
            raise
    else:
        cwb_data = BytesIO(response.read())
        content = cwb_data.read().decode("utf-8")

        for line in content.split("\n"):
            if line.startswith("mchirp:"):
                chirp_mass = float(line.split(":")[1].strip())
        output = {"chirp_mass": chirp_mass}
        return output


def process_cwb_pe(
    events: pd.DataFrame,
    gdb_server: str,
    max_workers: int = 15,
    raise_on_error: bool = False,
):
    func = partial(_cwb_wrapper, gdb_server=gdb_server)
    futures = parallelize(func, events, max_workers=max_workers)
    results = [None] * len(events)
    for future in tqdm(
        as_completed(futures),
        total=len(futures),
        desc="Processing cwb PE",
    ):
        idx = futures[future]
        try:
            results[idx] = future.result()
        except Exception as e:
            if raise_on_error:
                logging.error(f"Failed to process CWB for event {idx}")
                logging.error(traceback.format_exc())
                raise e
            else:
                logging.error(f"Failed to process CWB for event {idx}")
                results[idx] = None

    for key in ["chirp_mass"]:
        output = []
        for result in results:
            output.append(result[key] if result is not None else np.nan)
        events["cwb_" + key] = output

    return events


def _process_preferred_pe(
    row: pd.Series,
    gdb_server: str,
):
    """
    Process PE data for a single preferred event by routing to the
    appropriate pipeline processor.

    Args:
        row: Event row with preferred_pipeline column
        gdb_server: GraceDB server URL

    Returns:
        Dictionary with PE parameters or None if not processable
    """
    actual_pipeline = getattr(row, "preferred_pipeline", "unknown")
    gid = row.preferred_graceid

    if actual_pipeline == "aframe":
        return _process_posterior(gid, gdb_server)
    elif actual_pipeline == "cwb":
        return _process_cwb(gid, gdb_server)
    elif actual_pipeline in MATCHED_FILTERING_PIPELINES:
        return _process_coinc(gid, gdb_server)
    else:
        # Unknown or unsupported pipeline
        return None


def process_preferred_pe(
    events: pd.DataFrame,
    gdb_server: str,
    max_workers: int = 15,
    raise_on_error: bool = False,
) -> pd.DataFrame:
    """
    Process PE data for preferred events using parallelization.

    Args:
        events: DataFrame with preferred event data
        gdb_server: GraceDB server URL
        max_workers: Maximum number of worker processes

    Returns:
        DataFrame with PE parameters added
    """
    func = partial(_process_preferred_pe, gdb_server=gdb_server)
    futures = parallelize(func, events, max_workers=max_workers)
    results = [None] * len(events)

    for future in tqdm(
        as_completed(futures),
        total=len(futures),
        desc="Processing Preferred PE",
    ):
        idx = futures[future]
        try:
            results[idx] = future.result()
        except Exception as e:
            if raise_on_error:
                logging.error(
                    f"Failed to process preferred PE for event {idx}: {e}"
                )
                logging.error(traceback.format_exc())
                raise e
            else:
                logging.error(
                    f"Failed to process preferred PE for event {idx}: {e}"
                )
                results[idx] = None

    # Add results to dataframe based on the type of data returned
    for key in PE_KEYS + ["mass1", "mass2"]:
        output = []
        for result in results:
            output.append(
                result[key] if result is not None and key in result else np.nan
            )
        events[f"preferred_{key}"] = output

    return events


def process_pe(
    events: pd.DataFrame,
    pipeline: str,
    gdb_server: str,
    max_workers: int = 15,
    raise_on_error: bool = False,
) -> pd.DataFrame:
    """
    Process parameter estimation data for different pipelines including
    preferred events.

    Args:
        events: DataFrame with event data
        pipeline: Pipeline name (aframe, cwb, mbta, pycbc, gstlal, preferred)
        gdb_server: GraceDB server URL
        max_workers: Maximum number of worker processes
        raise_on_error: If True, raise exceptions with full traceback for
            debugging. If False, log errors and continue.

    Returns:
        DataFrame with PE parameters added
    """
    if pipeline == "preferred":
        events = process_preferred_pe(
            events, gdb_server, max_workers, raise_on_error
        )
    elif pipeline == "aframe":
        events = process_aframe_pe(
            events, gdb_server, max_workers, raise_on_error
        )
    elif pipeline == "cwb":
        events = process_cwb_pe(
            events, gdb_server, max_workers, raise_on_error
        )
    elif pipeline in MATCHED_FILTERING_PIPELINES:
        events = process_matched_filter_pe(
            events, gdb_server, pipeline, max_workers, raise_on_error
        )
    else:
        logging.warning(f"Unknown pipeline for PE processing: {pipeline}")

    return events
