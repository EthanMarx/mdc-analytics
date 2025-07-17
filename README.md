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
