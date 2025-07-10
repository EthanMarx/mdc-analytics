import json
import logging
import traceback
from functools import partial
from concurrent.futures import as_completed

import pandas as pd
import requests
from ligo.gracedb.rest import GraceDb
from tqdm.auto import tqdm

from .utils import parallelize

EM_BRIGHT_KEYS = ["HasMassGap", "HasNS", "HasRemnant", "HasSSM"]


def _process_embright(
    row: pd.Series,
    pipeline: str,
    gdb_server: str,
):
    # For preferred events, all pipelines can have EM bright data, so no
    # filtering needed
    gid = getattr(row, f"{pipeline}_graceid")
    if not gid:
        return None

    gdb = GraceDb(service_url=gdb_server, use_auth="scitoken")
    try:
        response = gdb.files(gid, "em_bright.json")
    except requests.exceptions.HTTPError as e:
        if e.response.status_code == 404:
            logging.warning(f"File 'em_bright.json' not found for event {gid}")
            return None
        else:
            raise
    else:
        em_bright = json.loads(response.read().decode("utf-8"))
        return em_bright


def process_embrights(
    events: pd.DataFrame,
    pipeline: str,
    gdb_server: str,
    max_workers: int = 15,
    raise_on_error: bool = False,
) -> pd.DataFrame:
    """
    Process EM bright data for different pipelines including preferred events.

    Args:
        events: DataFrame with event data
        pipeline: Pipeline name (aframe, cwb, mbta, pycbc, gstlal, preferred)
        gdb_server: GraceDB server URL
        max_workers: Maximum number of worker processes
        raise_on_error: If True, raise exceptions with full traceback for
            debugging. If False, log errors and continue.

    Returns:
        DataFrame with EM bright data added
    """
    func = partial(_process_embright, pipeline=pipeline, gdb_server=gdb_server)
    futures = parallelize(func, events, max_workers=max_workers)
    results = [None] * len(events)

    gids = getattr(events, f"{pipeline}_graceid").values
    for future in tqdm(
        as_completed(futures),
        total=len(futures),
        desc="Processing EM Bright",
    ):
        idx = futures[future]
        try:
            results[idx] = future.result()
        except Exception as e:
            if raise_on_error:
                logging.error(f"Failed to fetch em bright for {gids[idx]}")
                logging.error(traceback.format_exc())
                raise e
            else:
                results[idx] = None
                logging.error(f"Failed to fetch em bright for {gids[idx]}")

    for key in EM_BRIGHT_KEYS:
        output = []
        for result in results:
            output.append(
                result[key] if result is not None and key in result else False
            )
        events[f"{pipeline}_" + key] = output

    return events
