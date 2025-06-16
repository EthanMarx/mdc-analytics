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

Example configs are in `configs/crossmatch`

## Query Strain

