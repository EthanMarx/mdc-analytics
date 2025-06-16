from typing import Optional
from pathlib import Path
import logging
from jsonargparse import auto_cli
from .gracedb import query_gevents, cluster_gevents, process_coincs, process_skymaps, process_posteriors, process_embrights
from . import utils
import pandas as pd

def configure_logging():
    log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_format)
    logging.getLogger("scitokens").setLevel(logging.ERROR) 
    logging.getLogger("BAYESTAR").setLevel(logging.ERROR)
    logging.getLogger("matplotlib.texmanager").setLevel(logging.ERROR)

def crossmatch(
    outdir: Path,
    injection_file: Path,
    offset: float,
    flags: list[str], 
    pipelines: dict[str, str],
    dt: float = 0.5,
    ra_key: str =  "right_ascension",
    injection_time_key: str = "time_geocenter"
):
    """
    Crossmatch a ground truth "MDC" injection set with online analysis events submitted to GraceDB 

    Args:
        outdir: 
            Path to directory where output data products will be stored
        injection_file: 
            Path to the ground truth injection file in hdf5 format. The 
            file must have an `events` group, and be readable with pandas
            via `pd.read_hdf(injection_file, key="events)`.  
        offset:
            Offset that maps from ground truth injection times to 
            the corresponding time the injection occurs in the replay.
            The offset will also be used to account for corresponding 
            change in right ascension due to earths rotation
        flags:
            List of data quality flags to query and add to the dataframe.
            Will create boolean columns for each flag that indicates whether
            the injection occured during the requested flag.
        dt: 
            Time difference between injected and reported times 
            to consider an injection "recovered"
        ra_key: 
            Key in the dataframe corresponding to the injections right ascension
        injection_time_key:
            Key in the dataframe corresponding to the injections time at geocenter 
    """
    logging.info(f"Saving data to {outdir}")
    outdir.mkdir(parents=True, exist_ok=True)

    # construct a dataframe consisting of ground truth mdc events 
    # of interest, by making user requested filters, and removing events
    # that were not injected into science mode segments for the given ifos 
    logging.info(f"Reading MDC injection dataset from {injection_file}")
    events = pd.read_hdf(injection_file, key="events")
    events["time_geocenter_replay"] = events[injection_time_key] + offset 

    logging.info("Calculating ra offset and adding `right_ascension_offset` column to dataframe")
    events = utils.apply_skymap_offset(events, offset, ra_key)

    # add boolean columns that says if flags were active in science mode 
    logging.info("Appending data quality flag boolean columns")
    events = utils.append_data_quality_flags(
        events, 
        flags,
        events[injection_time_key].min(),
        events[injection_time_key].max()
    )

    # calculate start and stop times of injections in replay
    start = events.time_geocenter_replay.min() - 10
    stop = events.time_geocenter_replay.max() + 10

    # for each pipeline, query all gracedb uploads made  
    # from between the requested analysis `start` to `stop`
    for pipeline, server in pipelines.items():
        logging.info(f"Querying {pipeline} events between {start} and {stop} from {server}")
        pipeline_events = query_gevents(pipeline, server, start, stop)
        logging.info(f"Found {len(pipeline_events)} {pipeline} events") 
        
        logging.info(f"Clustering G events to most significant for each S event")
        pipeline_events = cluster_gevents(pipeline_events)
        logging.info(f"{len(pipeline_events)} after clustering to S events")

        # match pipeline events with the ground truth events
        # by assigning a corresponding gevent to the column.
        # if no aframe event was picked up, assign `None`
        logging.info(f"Crossmatching {pipeline} events with ground truth")
        events, injection_mask = utils.crossmatch_gevents(events, pipeline_events, pipeline, dt)

        logging.info(f"{sum(injection_mask)} events are within {dt}s of known injection")
        logging.info(f"{sum(~injection_mask)} events do not correspond with known injection")

        # filter for "noise" pipeline events 
        # so we can investigate them further
        pipeline_noise = pipeline_events[~injection_mask]
        pipeline_noise.to_hdf(outdir / f"{pipeline}_noise.hdf5", key="events")
    
        events = process_embrights(events, pipeline, server)
        
        # calculate searched area, vol, probs, etc.
        # and make relevant plots
        events = process_skymaps(events, pipeline, server) 

        if pipeline == "aframe":
            # query amplfi posterior files and 
            # create scatter plots
            events = process_posteriors(events, server)

        else:
            events = process_coincs(events, server, pipeline)

    
    # save master dataframe to disk
    logging.info(f"Saving dataframe with all events to {outdir / 'events.hdf5'}")
    events.to_hdf(outdir / "events.hdf5", key="events")

def main():
    configure_logging()
    auto_cli(crossmatch, as_positional=False)

if  __name__ == "__main__":
    main()