import h5py
import pandas as pd
from jsonargparse import auto_cli
from pathlib import Path
from typing import Literal, Optional
import logging
import concurrent
from tqdm.auto import tqdm
from . import utils
from functools import partial
import numpy as np


def configure_logging():
    log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_format)
    logging.getLogger("scitokens").setLevel(logging.ERROR)
    logging.getLogger("BAYESTAR").setLevel(logging.ERROR)
    logging.getLogger("matplotlib.texmanager").setLevel(logging.ERROR)


def query_strain(
    outdir: Path,
    injection_file: Path,
    ifos: list[str],
    channels: list[str],
    strain_data_dirs: dict[str, Path],
    sample_rate: float,
    psd_length: float,
    resample_method: Literal["lal", "gwpy"] = "gwpy",
    filters: Optional[list[tuple[str, float | str, float | str]]] = None,
    flags: Optional[list[str]] = None,
    injection_time_key: str = "time_geocenter",
    max_workers: int = 10,
):
    """
    Query strain data corresponding to known MDC injections
    from directories of GWF files into an hdf5 format
    """

    outdir.mkdir(exist_ok=True, parents=True)
    configure_logging()

    # read in injection file and apply filters
    events = pd.read_hdf(injection_file, key="events")
    events = utils.filter_events(events, filters)
    events = utils.append_data_quality_flags(
        events,
        flags,
        events[injection_time_key].min(),
        events[injection_time_key].max(),
        injection_time_key,
    )

    if flags is not None:
        logging.info(f"{len(events)} events before removing bad data quality")
        mask = np.ones(len(events), dtype=bool)
        for flag in flags:
            mask &= events[flag]

        events = events[mask]
        logging.info(f"{len(events)} events after removing bad data quality")
    fname_data = utils.parse_fnames(ifos, strain_data_dirs)
    strain = {ifo: [] for ifo in ifos}

    # Create a mapping to maintain original order
    results = {}

    func = partial(
        utils.query_event_strain,
        ifos=ifos,
        fname_data=fname_data,
        channels=channels,
        sample_rate=sample_rate,
        psd_length=psd_length,
        resample_method=resample_method,
    )
    with concurrent.futures.ProcessPoolExecutor(
        max_workers=max_workers
    ) as executor:
        future_to_index = {
            executor.submit(func, (i, event)): i
            for i, event in events.iterrows()
        }

        # process as they complete with progress bar
        for future in tqdm(
            concurrent.futures.as_completed(future_to_index), total=len(events)
        ):
            try:
                event_index, local_strain = future.result()
                results[event_index] = local_strain
                logging.info(f"Processed event {event_index}")
            except Exception as e:
                logging.error(f"Error processing event: {e}")
                raise

    # merge results in the original order
    for i in sorted(results.keys()):
        for ifo in ifos:
            strain[ifo].append(results[i][ifo])

    output = outdir / "strain.hdf5"
    with h5py.File(output, "w") as f:
        g = f.create_group("strain")
        for ifo, data in strain.items():
            g.create_dataset(ifo, data=data)

    logging.info(f"Writing strain data to {output}")
    events.to_hdf(output, key="parameters", mode="a")


def main():
    auto_cli(query_strain, as_positional=False)


if __name__ == "__main__":
    main()
