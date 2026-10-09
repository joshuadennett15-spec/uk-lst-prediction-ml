"""The fixed 2000 UK sample points (same draw as the notebook)."""
import ee
import pandas as pd

from ml2.config import MISSING, N_POINTS, POINT_SEED


def uk_points() -> ee.FeatureCollection:
    countries = ee.FeatureCollection("USDOS/LSIB_SIMPLE/2017")
    uk = countries.filter(ee.Filter.eq("country_na", "United Kingdom"))
    pts = ee.FeatureCollection.randomPoints(region=uk.geometry(), points=N_POINTS, seed=POINT_SEED)
    return pts.map(lambda f: f.set({
        "pid": f.id(),
        "lon": f.geometry().coordinates().get(0),
        "lat": f.geometry().coordinates().get(1),
    }))


def rows_to_frame(fc: ee.FeatureCollection) -> pd.DataFrame:
    """Download a small feature collection as a DataFrame, placeholders turned into NaN."""
    feats = fc.getInfo()["features"]
    df = pd.DataFrame([f["properties"] for f in feats])
    df["pid"] = df["pid"].astype(int)
    return df.replace(MISSING, float("nan"))
