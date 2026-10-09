"""Static per-point predictors: elevation and land-cover fractions within 500 m."""
import ee
import pandas as pd

from ml2.config import N_POINTS
from ml2.ee_points import rows_to_frame

LC_CLASSES = [10, 20, 30, 40, 50, 60, 70, 80, 90, 95, 100]
CHUNK = 250                                # points per request, keeps reduceRegions at 10 m within limits


def _chunks(points: ee.FeatureCollection):
    pts = points.toList(N_POINTS)
    for start in range(0, N_POINTS, CHUNK):
        yield ee.FeatureCollection(pts.slice(start, start + CHUNK))


def static_features(points: ee.FeatureCollection) -> pd.DataFrame:
    dem = ee.ImageCollection("COPERNICUS/DEM/GLO30").select("DEM").mosaic().rename("elevation")
    lc = ee.Image(ee.ImageCollection("ESA/WorldCover/v200").first()).select("Map")
    onehot = ee.Image.cat([lc.eq(c).rename(f"lc{c}") for c in LC_CLASSES])

    parts = []
    for chunk in _chunks(points):
        zones = chunk.map(lambda f: f.buffer(500))
        elev = rows_to_frame(dem.reduceRegions(zones, ee.Reducer.mean(), scale=30))
        elev = elev.rename(columns={"mean": "elevation"})
        frac = rows_to_frame(onehot.reduceRegions(zones, ee.Reducer.mean(), scale=10))
        parts.append(elev.drop(columns=["lon", "lat"]).merge(
            frac.drop(columns=["lon", "lat"]), on="pid", validate="one_to_one"))
    return pd.concat(parts, ignore_index=True)
