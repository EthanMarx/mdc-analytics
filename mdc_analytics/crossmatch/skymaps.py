import os
import logging
import traceback
from io import BytesIO
from pathlib import Path
from functools import partial
from concurrent.futures import as_completed
from typing import Optional

import numpy as np
import pandas as pd
import requests
from astropy.coordinates import SkyCoord
from astropy import units as u
from ligo.skymap.io.fits import read_sky_map
from ligo.skymap.bayestar import localize
from ligo.skymap.io import events
from ligo.skymap.postprocess.crossmatch import crossmatch
from ligo.gracedb.rest import GraceDb
from igwn_ligolw.utils import load_fileobj
from tqdm.auto import tqdm

from .utils import parallelize
from .pe import MATCHED_FILTERING_PIPELINES

CROSSMATCH_KEYS = [
    "searched_area",
    "searched_vol",
    "searched_prob",
    "searched_prob_vol",
    "searched_prob_dist",
]

PIPELINE_TO_SKYMAP = {
    "aframe": "amplfi.multiorder.fits",
    "gstlal": "bayestar.multiorder.fits",
    "spiir": "bayestar.multiorder.fits",
    "pycbc": "bayestar.multiorder.fits",
    "mbta": "bayestar.multiorder.fits",
    "cwb": "cwb.multiorder.fits",
}


def _get_gracedb_skymap(
    gdb: GraceDb,
    gid: str,
    pipeline: str,
) -> tuple[object, frozenset[str]]:
    """Download and parse skymap from GraceDB."""
    try:
        skymap_response = gdb.files(gid, PIPELINE_TO_SKYMAP[pipeline])
    except requests.exceptions.HTTPError as e:
        if e.response.status_code == 404:
            logging.warning(
                f"File '{PIPELINE_TO_SKYMAP[pipeline]}' not found for event "
                f"{gid}"
            )
            return None, None
        else:
            raise
    else:
        skymap_bytes = BytesIO(skymap_response.read())
        skymap = read_sky_map(skymap_bytes, moc=True)
        skymap_instruments = frozenset(skymap.meta["instruments"])
        return skymap, skymap_instruments


def _get_coinc_data(
    gdb: GraceDb,
    gid: str,
) -> tuple[object, frozenset[str]]:
    """Download and parse coinc.xml file for SNR timeseries."""
    try:
        coinc_response = gdb.files(gid, "coinc.xml")
    except requests.exceptions.HTTPError as e:
        if e.response.status_code == 404:
            logging.warning(f"File 'coinc.xml' not found for event {gid}")
            return None, None
        else:
            raise
    else:
        coinc_bytes = BytesIO(coinc_response.read())
        doc = load_fileobj(
            coinc_bytes, contenthandler=events.ligolw.ContentHandler
        )
        event_source = events.ligolw.open(doc, psd_file=doc, coinc_def=None)

        # Extract available instruments from SNR timeseries
        instruments = frozenset(
            [
                x.detector
                for x in event_source[list(event_source.keys())[0]].singles
            ]
        )
        return event_source, instruments


def _generate_bayestar_skymap(
    event_source: object,
    ifo_config: frozenset[str],
    available_instruments: frozenset[str],
) -> object:
    """Generate skymap using Bayestar for specific IFO configuration."""
    disable_detectors = available_instruments - ifo_config
    if (not disable_detectors) or (ifo_config <= available_instruments):
        return None

    # Disable detectors not in the requested configuration
    event_source = events.detector_disabled.open(
        event_source, disable_detectors, raises=False
    )
    (event,) = event_source.values()
    skymap = localize(event, waveform="IMRPhenomPv2", f_low=20)
    return skymap


def _crossmatch_skymap(
    skymap: object,
    coord: SkyCoord,
) -> object:
    """Perform crossmatching between skymap and injection coordinates."""
    return crossmatch(skymap, coord)


def _process_skymap_configs(
    gdb: GraceDb,
    gid: str,
    pipeline: str,
    coord: SkyCoord,
    bayestar_ifo_configs: Optional[list[frozenset[str]]],
) -> dict[frozenset[str], object]:
    """Process all requested IFO configurations for a single event."""

    # Process the uploaded GraceDB skymap
    gracedb_skymap, gracedb_instruments = _get_gracedb_skymap(
        gdb, gid, pipeline
    )
    if gracedb_skymap is None:
        # Return None for all requested configurations if no skymap found
        if bayestar_ifo_configs is None:
            return {gracedb_instruments: None}
        else:
            return dict.fromkeys(bayestar_ifo_configs)

    gracedb_result = _crossmatch_skymap(gracedb_skymap, coord)

    # If bayestar_ifo_configs is None, only return GraceDB skymap result
    if bayestar_ifo_configs is None:
        return {gracedb_instruments: gracedb_result}

    results = dict.fromkeys(bayestar_ifo_configs)
    results[gracedb_instruments] = gracedb_result

    # For matched filtering pipelines, generate additional Bayestar skymaps
    if pipeline in MATCHED_FILTERING_PIPELINES:
        event_source, available_instruments = _get_coinc_data(gdb, gid)

        # If coinc data not found, skip additional Bayestar generation
        if event_source is None:
            return {key: results.get(key) for key in bayestar_ifo_configs}

        # Verify that GraceDB skymap uses same instruments as coinc.xml
        assert available_instruments == gracedb_instruments

        # Generate skymaps for other requested IFO configurations
        for ifo_config in bayestar_ifo_configs:
            if ifo_config == gracedb_instruments:
                continue

            bayestar_skymap = _generate_bayestar_skymap(
                event_source, ifo_config, available_instruments
            )

            if bayestar_skymap is not None:
                result = _crossmatch_skymap(bayestar_skymap, coord)
                results[ifo_config] = result
            else:
                results[ifo_config] = None

    return {key: results[key] for key in bayestar_ifo_configs}


def _process_skymap(
    row: pd.Series,
    pipeline: str,
    gdb_server: str,
    bayestar_ifo_configs: Optional[list[frozenset[str]]],
):
    """Process skymap statistics for a single event."""
    # Set environment variable to speed up Bayestar
    os.environ["OMP_NUM_THREADS"] = "1"

    # For preferred events, get the actual pipeline
    actual_pipeline = pipeline
    if pipeline == "preferred":
        actual_pipeline = getattr(row, "preferred_pipeline", "unknown")
        if not actual_pipeline:
            return None

    gid = getattr(row, f"{pipeline}_graceid")
    if not gid:
        return None

    gdb = GraceDb(service_url=gdb_server, use_auth="scitoken")

    # Create coordinate object for the injection using pipeline-specific RA
    coord = SkyCoord(
        getattr(row, f"{pipeline}_right_ascension_offset") * u.rad,
        row.declination * u.rad,
        row.luminosity_distance * u.Mpc,
    )

    # Process all requested IFO configurations using the actual pipeline
    return _process_skymap_configs(
        gdb, gid, actual_pipeline, coord, bayestar_ifo_configs
    )


def process_skymaps(
    events: pd.DataFrame,
    pipeline: str,
    gdb_server: str,
    bayestar_ifo_configs: Optional[list[frozenset[str]]],
    max_workers: int = 15,
    raise_on_error: bool = False,
) -> pd.DataFrame:
    """
    Process skymap data for different pipelines including preferred events.

    Args:
        events: DataFrame with event data
        pipeline: Pipeline name (aframe, cwb, mbta, pycbc, gstlal, preferred)
        gdb_server: GraceDB server URL
        bayestar_ifo_configs: List of IFO configurations for Bayestar, or None
            to only use the GraceDB skymap IFO configuration
        max_workers: Maximum number of worker processes
        raise_on_error: If True, raise exceptions for debugging. If False,
            log and continue.

    Returns:
        DataFrame with skymap statistics added
    """
    func = partial(
        _process_skymap,
        gdb_server=gdb_server,
        pipeline=pipeline,
        bayestar_ifo_configs=bayestar_ifo_configs,
    )

    if bayestar_ifo_configs is not None:
        bayestar_ifo_configs = [
            cfg for cfg in bayestar_ifo_configs if cfg is not None
        ]
        if not bayestar_ifo_configs:
            logging.warning(
                "No valid bayestar_ifo_configs after filtering None entries"
            )
            return events

    futures = parallelize(func, events, max_workers=max_workers)

    if bayestar_ifo_configs is None:
        ifo_config_strs = []
        results = {}
    else:
        ifo_config_strs = [
            "".join(sorted(ifo_config)) for ifo_config in bayestar_ifo_configs
        ]
        results = {
            ifo_config: [None] * len(events) for ifo_config in ifo_config_strs
        }

    gids = getattr(events, f"{pipeline}_graceid").values
    for future in tqdm(
        as_completed(futures),
        total=len(futures),
        desc=f"Processing {pipeline} Skymaps",
    ):
        idx = futures[future]
        try:
            result = future.result()
        except Exception as e:
            if raise_on_error:
                logging.error(f"Failed to process skymap for {gids[idx]}")
                logging.error(traceback.format_exc())
                raise e
            else:
                logging.error(f"Failed to process skymap for {gids[idx]}")
                if bayestar_ifo_configs is None:
                    result = None
                else:
                    result = dict.fromkeys(ifo_config_strs)
                logging.error(e)

        if result is None:
            if bayestar_ifo_configs is None:
                continue
            for ifo_config in ifo_config_strs:
                results[ifo_config][idx] = None
        else:
            for ifo_config, res in result.items():
                if ifo_config is None:
                    logging.warning(
                        f"Skipping None ifo_config for event {gids[idx]}"
                    )
                    continue
                ifo_config_str = "".join(sorted(ifo_config))
                if ifo_config_str not in results:
                    results[ifo_config_str] = [None] * len(events)
                results[ifo_config_str][idx] = res

    for ifo_config in results.keys():
        ifo_config_str = "".join([c for c in ifo_config if not c.isdigit()])
        for key in CROSSMATCH_KEYS:
            events[f"{pipeline}_{key}_{ifo_config_str}"] = [
                getattr(result, key) if result is not None else np.nan
                for result in results[ifo_config]
            ]

    return events


def _process_skymap_offline(
    event: pd.Series,
    kde: bool = False,
):
    if event.aframe_offline_url is None:
        return None

    skymap_fname = "amplfi.multiorder.fits"
    if kde:
        skymap_fname = "amplfi.multiorder.fits,0"
    skymap = Path(event.aframe_offline_url).parent / skymap_fname

    skymap = read_sky_map(skymap, moc=True)
    coord = SkyCoord(
        ra=event.right_ascension * u.rad,
        dec=event.declination * u.rad,
        distance=event.luminosity_distance * u.Mpc,
    )

    result = crossmatch(skymap, coord)
    return result
