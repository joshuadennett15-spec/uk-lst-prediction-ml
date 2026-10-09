"""Constants shared by the Earth Engine sampling code."""
import os
from pathlib import Path

EE_PROJECT = os.environ.get("EE_PROJECT", "mlprojects-510908")
N_POINTS = 2000
POINT_SEED = 42
YEARS = tuple(range(2003, 2024))          # 21 summers
SUMMER_START = "06-01"                    # composites starting from here...
SUMMER_END = "09-01"                      # ...up to (not including) here
MISSING = -9999                           # placeholder so every point returns a row

MODIS_SCALE = 926.6254331383326           # MODIS native pixel size (m); never sample at a rounded 1000
ERA_SCALE = 11131.949079327358            # ERA5-Land native pixel size (m)

RAW_DIR = Path("data/raw")
