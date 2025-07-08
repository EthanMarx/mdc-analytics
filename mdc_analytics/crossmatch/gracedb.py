import h5py
import json
import numpy as np
import pandas as pd
from io import BytesIO
from ligo.skymap.io.fits import read_sky_map
from ligo.skymap.bayestar import localize
from ligo.skymap.io import events
from ligo.skymap.postprocess.crossmatch import crossmatch
from astropy.coordinates import SkyCoord
from astropy import units as u
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm.auto import tqdm
from typing import Optional
from functools import partial
from igwn_ligolw import lsctables
from igwn_ligolw import utils as ligolw_utils
from ligo.gracedb.rest import GraceDb
import logging
from pathlib import Path
from ligo.lw.utils import load_fileobj

CROSSMATCH_KEYS = ["searched_area", "searched_vol", "searched_prob", "searched_prob_vol"]
PE_KEYS = ["chirp_mass", "luminosity_distance"]
EM_BRIGHT_KEYS = ["HasMassGap", "HasNS", "HasRemnant", "HasSSM"]

PIPELINE_TO_SKYMAP = {
    "aframe": "amplfi.multiorder.fits",
    "gstlal": "bayestar.multiorder.fits",
    "pycbc": "bayestar.multiorder.fits",
    "mbta": "bayestar.multiorder.fits",
    "cwb": "cwb.multiorder.fits"

}

# mapping from gevent columns 
# to datatype for storage in pandas df
GEVENT_COLUMNS = {
    'graceid': 'str',
    'gpstime': 'float64',
    'superevent': 'str',
    'search': 'str',
    'instruments': 'str',
    'far': 'float64'
}

def query_gevents(
    pipeline: str,
    gdb_server: str, 
    start: float, 
    stop: float,
    search: str = None,
) -> pd.DataFrame:
    
    """
    Query gracedb events between start and stop for a given pipeline from a specific gracedb server
    """
    client = GraceDb(service_url=gdb_server, use_auth="scitoken")
    if pipeline != "preferred":
        query = f"{pipeline} gpstime: {start} .. {stop} "
        if search is not None:    
            query += f"search: {search} "
        func = client.events
    else:
        query = f"gpstime: {start} .. {stop} " 
        func = client.superevents

    response = func(query)

    if pipeline == "preferred":
        pipeline_events = list(response)
        pipeline_events = pd.DataFrame(pipeline_events) 
        query = " ".join(pipeline_events.preferred_event.values)
        response = client.events(query)
        
    pipeline_events = list(response)
    pipeline_events = pd.DataFrame(pipeline_events)
    pipeline_events = pipeline_events[list(GEVENT_COLUMNS.keys())]
    pipeline_events = pipeline_events.astype(GEVENT_COLUMNS)
    return pipeline_events

def cluster_gevents(gevents: pd.DataFrame) -> pd.DataFrame:
    # keep only the most significant event per superevent,
    # where significant is 

    # TODO: this handles case when aframe didn't 
    # have superevents yet - remove 
    none_mask: np.ndarray = gevents['superevent'].values == "None"
    none_events = gevents[none_mask] 
    
    non_none_events = gevents[~none_mask]

    if not non_none_events.empty:
        clustered = non_none_events.groupby('superevent', group_keys=False).apply(
            lambda x: x.loc[x['far'].idxmin()],
            include_groups=True,
        ).reset_index(drop=True)
    else:
        clustered = pd.DataFrame()

    result = pd.concat([none_events, clustered], ignore_index=True)

    return result 

_global_pool = None

def get_pool(max_workers=None):
    global _global_pool
    if _global_pool is None:
        logging.info("Initializing pool")
        _global_pool = ProcessPoolExecutor(max_workers=max_workers)
    return _global_pool

def shutdown_global_pool():
    global _global_pool
    if _global_pool is not None:
        _global_pool.shutdown()
        _global_pool = None

def parallelize(
    func: callable,
    events: pd.DataFrame,
    max_workers=None,

):
    executor = get_pool(max_workers)
    future_to_index = {
        executor.submit(func, row): i
        for i, (_, row) in enumerate(events.iterrows())
    }
    return future_to_index 

def _process_skymap(
    row: pd.Series,
    pipeline: str,
    gdb_server: str,
    bayestar_ifo_configs: list[frozenset[str]],
):

    gid = getattr(row, f"{pipeline}_graceid") 
    if gid is None:
        return None

    gdb = GraceDb(service_url=gdb_server, use_auth="scitoken")
    skymap_response = gdb.files(gid, PIPELINE_TO_SKYMAP[pipeline]) 
    skymap_bytes = BytesIO(skymap_response.read())
    skymap = read_sky_map(skymap_bytes, moc=True)
    instruments = frozenset(skymap.meta["instruments"])

    coord = SkyCoord(
        row.right_ascension_offset * u.rad, 
        row.declination * u.rad, 
        row.luminosity_distance * u.Mpc,
    )

    coinc_response = gdb.files(gid, "coinc.xml")
    coinc_bytes = BytesIO(coinc_response.read())
    doc = load_fileobj(BytesIO(coinc_bytes), contenthandler=events.ligolw.ContentHandler)
    event_source = events.ligolw.open(doc, psd_file=doc, coinc_def=None)

    result = crossmatch(skymap, coord)
    results = {} 
    results[instruments] = result
    
    for ifo_config in bayestar_ifo_configs:
        if ifo_config == instruments:
            continue

        disable_detectors = instruments - ifo_config
        if disable_detectors:
            # set raises to False to avoid raising an error
            # if the detector is not in the coinc file
            event_source = events.detector_disabled.open(
                event_source, disable_detectors, raises=False
            )
            event, = event_source.values()
            skymap = localize(event, waveform="IMRPhenomPv2", f_low=20)

            result = crossmatch(skymap, coord)
            results[ifo_config] = result
        else:
            results[ifo_config] = None
    return results

def process_skymaps(
    events: pd.DataFrame,
    pipeline: str,
    gdb_server: str,
    bayestar_ifo_configs: list[frozenset[str]]
) -> pd.DataFrame:

    func = partial(_process_skymap, gdb_server=gdb_server, pipeline=pipeline, bayestar_ifo_configs=bayestar_ifo_configs)
    futures = parallelize(func, events) 
    ifo_config_strs = ["".join([x[0] for x in sorted(ifo_config)]) for ifo_config in bayestar_ifo_configs]
    results = {ifo_config: [None] * len(events) for ifo_config in ifo_config_strs}

    gids = getattr(events, f"{pipeline}_graceid").values
    for future in tqdm(
        as_completed(futures),
        total=len(futures),
        desc="Processing Skymaps",
    ):
        idx = futures[future]
        try:
            result = future.result()
        except Exception as e:
            logging.info(f"Failed to process skymap for {gids[idx]}")
            result = {ifo_config: None for ifo_config in ifo_config_strs}
            raise e
        if result is None: 
            for ifo_config in ifo_config_strs:
                results[ifo_config][idx] = None 
        else:
            for ifo_config, res in result.items():
                results[ifo_config][idx] = res

    # append bayestar statistics for each inteferometer combination
    for ifo_config in results.keys():
        ifo_config_str = "".join([x[0] for x in sorted(ifo_config)])
        for key in CROSSMATCH_KEYS:
            events[f"{pipeline}_{key}_{ifo_config_str}"] = [getattr(result, key) if result else None for result in results[ifo_config_str]]

    return events

def _process_posterior(
    row: pd.Series,
    gdb_server: str,
):
    if row.aframe_graceid is None:
        return None

    gdb = GraceDb(service_url=gdb_server, use_auth="scitoken") 
    response = gdb.files(row.aframe_graceid, 'amplfi.posterior_samples.hdf5') 
    posterior = BytesIO(response.read()) 

    output = {}
    with h5py.File(posterior) as f:
        for key in PE_KEYS: 
            output[key] = np.median(f["posterior_samples"][key][:])
    return output 



def process_posteriors(
    events: pd.DataFrame,
    gdb_server: str
):

    func = partial(_process_posterior, gdb_server=gdb_server)
    futures = parallelize(func, events)
    results = [None] * len(events)
    for future in tqdm(
        as_completed(futures),
        total=len(futures),
        desc="Processing Posteriors",
    ):
        idx = futures[future]
        try:
            results[idx] = future.result()
        except Exception:
            logging.info(f"Failed to query posterior for {events.aframe_graceid.values[idx]}")
            results[idx] = None
     
    for key in PE_KEYS:
        output = []
        for result in results:
            output.append(result[key] if result is not None else None)
        events["aframe_" + key] = output

    return events

def _process_coinc(
    row: pd.Series,
    pipeline: str,
    gdb_server: str,
):
    gid = getattr(row, f"{pipeline}_graceid")
    if gid is None:
        return None

    gdb = GraceDb(service_url=gdb_server, use_auth="scitoken") 
    response = gdb.files(gid, 'coinc.xml') 
    coinc_data = BytesIO(response.read())
    doc = ligolw_utils.load_fileobj(coinc_data)
    inspiral_table = lsctables.SnglInspiralTable.get_table(doc)

    output = {}
    for row in inspiral_table:
        output["chirp_mass"] = row.mchirp
        output["mass1"] = row.mass1
        output["mass2"] = row.mass2
    return output

def process_coincs(
    events: pd.DataFrame,
    gdb_server: str,
    pipeline: str
):
    func = partial(_process_coinc, pipeline=pipeline, gdb_server=gdb_server)
    futures = parallelize(func, events)
    results = [None] * len(events)
    gids = getattr(events, f"{pipeline}_graceid").values
    for future in tqdm(
        as_completed(futures),
        total=len(futures),
        desc="Processing Coincs",
    ):
        idx = futures[future]
        try:
            results[idx] = future.result()
        except Exception:
            logging.info(f"Failed to parse coinc for {gids[idx]}")
            results[idx] = None 
    for key in ["chirp_mass", "mass1", "mass2"]:
        output = []
        for result in results:
            output.append(result[key] if result is not None else None)
        events[f"{pipeline}_" + key] = output

    return events


def _process_cwb(
    row: pd.Series,
    gdb_server: str,
):
    gid = getattr(row, f"cwb_graceid")
    if gid is None:
        return None

    gdb = GraceDb(service_url=gdb_server, use_auth="scitoken") 
    response = gdb.files(gid, 'trigger.txt')     
    cwb_data = BytesIO(response.read())
    content = cwb_data.read().decode('utf-8') 
            
    for line in content.split('\n'):
        if line.startswith('mchirp:'):
            chirp_mass = float(line.split(':')[1].strip())
    output = {"chirp_mass": chirp_mass}
    return output

def process_cwb(
    events: pd.DataFrame,
    gdb_server: str,
):
    func = partial(_process_cwb, gdb_server=gdb_server)
    futures = parallelize(func, events)
    results = [None] * len(events)
    for future in tqdm(
        as_completed(futures),
        total=len(futures),
        desc="Processing CWB",
    ):
        idx = futures[future]
        results[idx] = future.result()
     
    for key in ["chirp_mass"]:
        output = []
        for result in results:
            output.append(result[key] if result is not None else None)
        events[f"cwb_" + key] = output

    return events


def _process_embright(
    row: pd.Series,
    pipeline: str,
    gdb_server: str,
):
    gid = getattr(row, f"{pipeline}_graceid")
    if gid is None:
        return None

    gdb = GraceDb(service_url=gdb_server, use_auth="scitoken") 
    response = gdb.files(gid, 'em_bright.json') 
    em_bright = json.loads(response.read().decode("utf-8"))
    return em_bright

def process_embrights(
    events: pd.DataFrame,
    pipeline: str,
    gdb_server: str,
):
    func = partial(_process_embright, pipeline=pipeline, gdb_server=gdb_server)
    futures = parallelize(func, events)
    results = [None] * len(events)
    for future in tqdm(
        as_completed(futures),
        total=len(futures),
        desc="Processing EM Bright",
    ):
        idx = futures[future]
        try:
            results[idx] = future.result()
        except Exception as e:
            results[idx] = None
            logging.info(f"Failed to fetch em bright for {idx}")
     
    for key in EM_BRIGHT_KEYS: 
        output = []
        for result in results:
            output.append(result[key] if result is not None and key in result else None)
        events[f"{pipeline}_" + key] = output

    return events

def _process_skymap_offline(
    event: pd.Series, 
    kde: bool = False,
):
    if event.aframe_offline_url is None:
        return None

    skymap_fname = 'amplfi.multiorder.fits' 
    if kde:
        skymap_fname = 'amplfi.multiorder.fits,0'
    skymap = Path(event.aframe_offline_url).parent / skymap_fname 

    skymap = read_sky_map(skymap, moc=True)
    coord = SkyCoord(
        ra=event.right_ascension * u.rad, 
        dec=event.declination * u.rad, 
        distance=event.luminosity_distance * u.Mpc 
    )

    result = crossmatch(skymap, coord)
    return result