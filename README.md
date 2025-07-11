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

Then, 

```
mdc-crossmatch --config /path/to/config.yaml
```

Example configs for the LVKs O3 MDC and O4 LLPIC are in the `configs/crossmatch` directory.


## Query Strain


## Data

The `./data` directory contains the injection files for the O3 MDC and O4 MDC for convenience
