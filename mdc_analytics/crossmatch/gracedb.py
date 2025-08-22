import pandas as pd
import numpy as np
from ligo.gracedb.rest import GraceDb
from itertools import batched
from tqdm.auto import tqdm

# mapping from gevent columns to (datatype, default_value) for pandas df
GEVENT_COLUMNS = {
    "graceid": ("string", ""),
    "gpstime": ("float64", float("nan")),
    "superevent": ("string", ""),
    "search": ("string", ""),
    "instruments": ("string", ""),
    "far": ("float64", float("nan")),
}

SUPPORTED_PIPELINES = {"mbta", "spiir", "cwb", "aframe", "gstlal", "pycbc"}


def query_gevents(
    pipeline: str,
    gdb_server: str,
    start: float,
    stop: float,
    search: str = None,
) -> pd.DataFrame:
    """
    Query gracedb events between start and stop for a given pipeline from a
    specific gracedb server
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

    preferred_pipelines = {}  # Map from graceid to actual pipeline

    if pipeline == "preferred":
        superevents = list(response)
        superevents_df = pd.DataFrame(superevents)

        # Extract preferred pipeline for each superevent
        for _, superevent in superevents_df.iterrows():
            preferred_event_id = superevent.get("preferred_event")
            if preferred_event_id and "preferred_event_data" in superevent:
                preferred_data = superevent["preferred_event_data"]
                preferred_pipeline = preferred_data["pipeline"].lower()
                if preferred_pipeline in SUPPORTED_PIPELINES:
                    preferred_pipelines[preferred_event_id] = (
                        preferred_pipeline
                    )

        response = []
        for batch in tqdm(
            batched(superevents_df.preferred_event.values, 1000),
            total=len(superevents_df) // 1000 + 1,
        ):
            gevent = client.events(" ".join(batch))
            response.extend(gevent)

    pipeline_events = list(response)
    pipeline_events = pd.DataFrame(pipeline_events)
    pipeline_events = pipeline_events[list(GEVENT_COLUMNS.keys())]
    # Extract dtypes from GEVENT_COLUMNS tuples
    dtypes = {col: dtype for col, (dtype, _) in GEVENT_COLUMNS.items()}
    pipeline_events = pipeline_events.astype(dtypes)

    # Add preferred_pipeline column for preferred events
    if pipeline == "preferred":
        pipeline_events["preferred_pipeline"] = pipeline_events["graceid"].map(
            lambda gid: preferred_pipelines.get(gid, "")
        )

    return pipeline_events


def cluster_gevents(gevents: pd.DataFrame) -> pd.DataFrame:
    # keep only the most significant event per superevent,
    # where significant is

    # TODO: this handles case when aframe didn't
    # have superevents yet - remove
    none_mask: np.ndarray = gevents["superevent"].values == "None"
    none_events = gevents[none_mask]

    non_none_events = gevents[~none_mask]

    if not non_none_events.empty:
        clustered = (
            non_none_events.groupby("superevent", group_keys=False)
            .apply(
                lambda x: x.loc[x["far"].idxmin()],
                include_groups=False,
            )
            .reset_index(drop=True)
        )
    else:
        clustered = pd.DataFrame()

    result = pd.concat([none_events, clustered], ignore_index=True)

    return result
