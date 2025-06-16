import h5py
import numpy as np
import pandas as pd
from io import BytesIO
from ligo.skymap.io.fits import read_sky_map
from ligo.skymap.postprocess.crossmatch import crossmatch
from astropy.coordinates import SkyCoord
from astropy import units as u
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm.auto import tqdm
from functools import partial
from igwn_ligolw import lsctables
from igwn_ligolw import utils as ligolw_utils
from ligo.gracedb.rest import GraceDb
import logging

CROSSMATCH_KEYS = ["searched_area", "searched_vol", "searched_prob", "searched_prob_vol"]
PE_KEYS = ["chirp_mass", "luminosity_distance"]

PIPELINE_TO_SKYMAP = {
    "aframe": "amplfi.fits",
    "gstlal": "bayestar.multiorder.fits"
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
    stop: float
) -> pd.DataFrame:
    
    """
    Query gracedb events between start and stop for a given pipeline from a specific gracedb server
    """
    client = GraceDb(service_url=gdb_server, use_auth="scitoken")
    query = f"{pipeline} gpstime: {start} .. {stop}"
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


def parallelize(
    func: callable,
    events: pd.DataFrame,
    max_workers=None,

):
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        future_to_index = {
            executor.submit(func, row): i
            for i, (_, row) in enumerate(events.iterrows())
        }
    return future_to_index 

def _process_skymap(
    row: pd.Series,
    pipeline: str,
    gdb_server: str,
):

    gid = getattr(row, f"{pipeline}_graceid") 
    if gid is None:
        return None

    gdb = GraceDb(service_url=gdb_server, use_auth="scitoken")
    response = gdb.files(gid, PIPELINE_TO_SKYMAP[pipeline]) 
    skymap_bytes = BytesIO(response.read())
    skymap = read_sky_map(skymap_bytes, moc=True)
    coord = SkyCoord(
        row.right_ascension_offset * u.rad, 
        row.declination * u.rad, 
        row.luminosity_distance * u.Mpc,
    )
    result = crossmatch(skymap, coord)
    return result

def process_skymaps(
    events: pd.DataFrame,
    pipeline: str,
    gdb_server: str,
) -> pd.DataFrame:

    func = partial(_process_skymap, gdb_server=gdb_server, pipeline=pipeline)
    futures = parallelize(func, events) 
    results = [None] * len(events)

    gids = getattr(events, f"{pipeline}_graceid").values
    for future in tqdm(
        as_completed(futures),
        total=len(futures),
        desc="Processing Skymaps",
    ):
        idx = futures[future]
        try:
            results[idx] = future.result()
        except Exception:
            logging.info(f"Failed to process skymap for {gids[idx]}")
            results[idx] = None

    for key in CROSSMATCH_KEYS:
        events[f"{pipeline}_{key}"] = [getattr(result, key) if result else None for result in results]

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
    for future in tqdm(
        as_completed(futures),
        total=len(futures),
        desc="Processing Coincs",
    ):
        idx = futures[future]
        results[idx] = future.result()
     
    for key in ["chirp_mass", "mass1", "mass2"]:
        output = []
        for result in results:
            output.append(result[key] if result is not None else None)
        events[f"{pipeline}_" + key] = output

    return events