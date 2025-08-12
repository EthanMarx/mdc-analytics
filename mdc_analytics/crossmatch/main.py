from typing import Optional, Literal
from pathlib import Path
import logging
from dataclasses import dataclass
from jsonargparse import auto_cli
from .gracedb import (
    query_gevents,
    cluster_gevents,
)
from .skymaps import process_skymaps
from .pe import process_pe
from .embrights import process_embrights
from . import utils
from .utils import shutdown_global_pool
import pandas as pd

PIPELINE = Literal[
    "aframe", "cwb", "gstlal", "pycbc", "mbta", "spiir", "preferred"
]


@dataclass
class PipelineConfig:
    """Configuration for a single pipeline."""

    name: PIPELINE
    server: str
    offset: float
    search: Optional[str] = None


def configure_logging():
    import warnings

    log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_format)
    logging.getLogger("scitokens").setLevel(logging.ERROR)
    logging.getLogger("BAYESTAR").setLevel(logging.ERROR)
    logging.getLogger("matplotlib.texmanager").setLevel(logging.ERROR)
    # Silence X.509 certificate expiration warnings
    warnings.filterwarnings(
        "ignore",
        message="Failed to validate.*X.509 certificate has less than.*seconds remaining",  # noqa: E501
    )
    warnings.filterwarnings(
        "ignore",
        message=".*add.*newdoc_ufunc is deprecated.*",
        category=DeprecationWarning,
        module="ligo.skymap.util.numpy",
    )


def crossmatch(
    outdir: Path,
    injection_file: Path,
    flags: list[str],
    pipelines: list[PipelineConfig],
    dt: float = 0.5,
    bayestar_ifo_configs: Optional[list[frozenset[str]]] = None,
    ra_key: str = "right_ascension",
    injection_time_key: str = "time_geocenter",
    max_workers: int = 15,
    filters: Optional[list[tuple[str, float | str, float | str]]] = None,
    raise_on_error: bool = False,
):
    """
    Crossmatch a ground truth "MDC" injection set with online analysis
    events submitted to GraceDB

    Args:
        outdir:
            Path to directory where output data products will be stored
        injection_file:
            Path to the ground truth injection file in hdf5 format. The
            file must have an `events` group, and be readable with pandas
            via `pd.read_hdf(injection_file, key="events)`.
        flags:
            List of data quality flags to query and add to the dataframe.
            Will create boolean columns for each flag that indicates whether
            the injection occured during the requested flag.
        pipelines:
            A list of PipelineConfig objects, each specifying a pipeline name,
            GraceDB server URL, time offset for that pipeline, and optional
            search filter (e.g. "AllSky")
        dt:
            Time difference between injected and reported times
            to consider an injection "recovered"
        bayestar_ifo_configs:
            Interferometer configurations for which to run bayestar. For each
            configuration
            that doesn't correspond to the gevents skymap, if the coinc.xml has
            the appropiate timeseries for that configuration,
            will run bayestar and calc statistics for that configuration
        ra_key:
            Key in the dataframe corresponding to the injections right
            ascension
        injection_time_key:
            Key in the dataframe corresponding to the injections time at
            geocenter
        max_workers:
            Maximum number of worker processes for parallel processing
        filters:
            Optional list of tuples (column_name, min_value, max_value) to
            filter events
        raise_on_error:
            If True, raise exceptions with full traceback for debugging. If
            False, log errors and continue.
    """
    if bayestar_ifo_configs is not None:
        bayestar_ifo_configs = [frozenset(s) for s in bayestar_ifo_configs]
        config_strings = [
            "{" + ", ".join(sorted(config)) + "}"
            for config in bayestar_ifo_configs
        ]
        logging.info(
            f"Analysing detector configurations {config_strings} with bayestar"
        )
    else:
        logging.info(
            "No bayestar_ifo_configs specified, only using GraceDB skymap"
        )

    logging.info(f"Saving data to {outdir}")
    outdir.mkdir(parents=True, exist_ok=True)

    # construct a dataframe consisting of ground truth mdc events
    # of interest, by making user requested filters, and removing events
    # that were not injected into science mode segments for the given ifos
    logging.info(f"Reading MDC injection dataset from {injection_file}")
    events = pd.read_hdf(injection_file, key="events")
    logging.info(f"Loaded {len(events)} initial events from injection file")
    events = utils.filter_events(events, filters)
    logging.info(f"After filtering: {len(events)} events remaining")
    events["luminosity_distance"] = events["distance"]

    # add boolean columns that says if flags were active in science mode
    logging.info("Appending data quality flag boolean columns")
    events = utils.append_data_quality_flags(
        events,
        flags,
        events[injection_time_key].min(),
        events[injection_time_key].max(),
        injection_time_key,
    )

    # for each pipeline, query all gracedb uploads made
    # from between the requested analysis `start` to `stop`
    for pipeline_config in pipelines:
        pipeline = pipeline_config.name
        server = pipeline_config.server
        search = pipeline_config.search
        offset = pipeline_config.offset

        # Create pipeline-specific time and RA offset columns
        logging.info(
            f"Calculating {pipeline} time and RA offsets with offset {offset}"
        )
        events[f"{pipeline}_time_geocenter_replay"] = (
            events[injection_time_key] + offset
        )
        events = utils.apply_skymap_offset(
            events, offset, ra_key, f"{pipeline}_right_ascension_offset"
        )

        # Calculate start and stop times for this pipeline
        start = events[f"{pipeline}_time_geocenter_replay"].min() - 10
        stop = events[f"{pipeline}_time_geocenter_replay"].max() + 10

        logging.info(
            f"Querying {pipeline}{' ' + search if search else ''} events "
            f"between {start} and {stop} from {server}"
        )
        pipeline_events = query_gevents(pipeline, server, start, stop, search)
        logging.info(f"Found {len(pipeline_events)} {pipeline} events")
        logging.info(
            "Clustering G events to most significant for each S event"
        )
        pipeline_events = cluster_gevents(pipeline_events)
        logging.info(f"{len(pipeline_events)} after clustering to S events")

        # match pipeline events with the ground truth events
        # by assigning a corresponding gevent to the column.
        # if no aframe event was picked up, assign `None`
        logging.info(f"Crossmatching {pipeline} events with ground truth")
        events, injection_mask = utils.crossmatch_gevents(
            events, pipeline_events, pipeline, dt
        )

        logging.info(
            f"{sum(injection_mask)} events are within {dt}s of known injection"
        )
        logging.info(
            f"{sum(~injection_mask)} events do not correspond with known "
            f"injection"
        )

        # filter for "noise" pipeline events
        # so we can investigate them further
        pipeline_noise = pipeline_events[~injection_mask]
        pipeline_noise.to_hdf(outdir / f"{pipeline}_noise.hdf5", key="events")

        # process parameter estimation data
        events = process_pe(
            events,
            pipeline,
            server,
            max_workers=max_workers,
            raise_on_error=True,
        )

        # calculate searched area, vol, probs, etc.
        # also, for matched filtering pipelines that
        # have coinc.xml files, optionally reanalyze
        # with different bayestar detector configurations
        events = process_skymaps(
            events,
            pipeline,
            server,
            bayestar_ifo_configs=bayestar_ifo_configs,
            max_workers=max_workers,
            raise_on_error=True,
        )

        # query embright probabilities
        events = process_embrights(
            events,
            pipeline,
            server,
            max_workers=max_workers,
            raise_on_error=True,
        )

    logging.info("Shutting down pool")
    shutdown_global_pool()

    # Fix data types before saving to avoid HDF5 compatibility issues
    if "approximant" in events.columns:
        events["approximant"] = events["approximant"].astype("string")

    # save master dataframe to disk
    logging.info(
        f"Saving dataframe with all events to {outdir / 'events.hdf5'}"
    )
    events.to_hdf(outdir / "events.hdf5", key="events")
    events.to_csv(outdir / "events.csv")


def main():
    configure_logging()
    auto_cli(crossmatch, as_positional=False)


if __name__ == "__main__":
    main()
