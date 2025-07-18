#!/bin/bash
tar -xzf mdc-analytics_sandbox.tar.gz

# Note: the condor.sub file should have the --config and --injection_file paths specified
# using `transfer_input_files`, 
apptainer exec --fakeroot mdc-analytics_sandbox/ mdc-crossmatch --config o3-mdc.yaml --injection_file o3-mdc-injections.hdf5 --outdir ./ --max_workers 15
