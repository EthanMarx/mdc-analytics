# mdc-analytics
Tools for analyzing GraceDb events corresponding to MDC injection campaigns

## Installation

```
pip install .
```

or

```
uv sync
```

## Crossmatch
First, authenticate to gracedb and segment data base via scitokens

```
htgettoken -a vault.ligo.org -i igwn --scopes gracedb.read,dqsegdb.read
```

It may be useful to set 

```
export IGWN_AUTH_UTILS_FIND_X509=0
``` 

to avoid igwn_auth_utils searching for X509 credentials.

Then, 

```
mdc-crossmatch --config /path/to/config.yaml
```

Example configs for the LVKs O3 MDC and O4 LLPIC are in the `configs/crossmatch` directory.


## Query Strain


## Data

The `./data` directory contains the injection files for the O3 MDC and O4 MDC for convenience


## Running on OSG with Condor

For use with condor, you must run on the OSG due to reliance on gracedb.
To do so, you must build the apptainer image as a sandbox, that will
be transferred and unzipped on the execute node

```
apptainer build mdc-analytics.sif apptainer.def
```

```
apptainer build --sandbox mdc-analytics_sandbox/ mdc-analytics.sif
```

```
tar -czf mdc-analytics_sandbox.tar.gz mdc-analytics_sandbox/
```

Now, you can submit via condor - see the condor.sub and executable.sh files
