#!/bin/bash
cd /home/ethan.marx/projects/mdc-analytics
uv run mdc-crossmatch --config configs/crossmatch/o3-mdc.yaml --max_workers 15 
