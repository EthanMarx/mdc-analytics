import pandas as pd
from jsonargparse import auto_cli
from gwpy.timeseries import TimeSeries
from pathlib import Path
from typing import Literal
import logging
import concurrent
from tqdm.auto import tqdm
import utils
from functools import partial

def query_strain(
    outdir: Path,
    injection_file: Path, 
    ifos: list[str],
    channels: list[str], 
    strain_data_dir: Path,
    sample_rate: float, 
    psd_length: float, 
    resample_method: Literal["lal", "gwpy"] = "gwpy", 
    max_workers: int = 10
):
    """
    Query strain data corresponding to known MDC injections from directories of GWF files 
    into an hdf5 format
    """

    events = pd.read_hdf(injection_file, key="events")
    fname_data = utils.parse_fnames(ifos, strain_data_dir) 
    
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
        resample_method=resample_method
    ) 
    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        future_to_index = {
            executor.submit(func, (i, event)): i 
            for i, event in events.iterrows()
        }
        
        # process as they complete with progress bar
        for future in tqdm(concurrent.futures.as_completed(future_to_index), total=len(events)):
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
    
    return strain


def main():
    auto_cli(query_strain, as_positional=False)

if __name__ == "__main__":
    main()