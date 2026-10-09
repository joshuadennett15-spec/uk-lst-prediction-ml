"""Per-composite predictor sampling: MODIS bands on the MODIS grid, ERA5 bands on the ERA5 grid."""
import ee

from ml2.config import ERA_SCALE, MISSING, MODIS_SCALE

LST_ALL = ee.ImageCollection("MODIS/061/MOD11A2")
NDVI_ALL = ee.ImageCollection("MODIS/061/MOD13A2").select("NDVI")
ERA = ee.ImageCollection("ECMWF/ERA5_LAND/DAILY_AGGR")


def lst_celsius(img: ee.Image) -> ee.Image:
    return img.select("LST_Day_1km").multiply(0.02).subtract(273.15)


def sat_lags(t: ee.Date) -> ee.Image:
    """Previous-composite LST (+ its QC) and the latest NDVI at least 15 days old."""
    prev = ee.Image(LST_ALL.filterDate(t.advance(-8, "day"), t).first())
    lst_prev = lst_celsius(prev).rename("lst_prev")
    qc_prev = prev.select("QC_Day").rename("qc_prev")
    nd = ee.Image(NDVI_ALL.filterDate(t.advance(-60, "day"), t.advance(-15, "day"))
                  .sort("system:time_start", False).first())
    ndvi_lag = nd.multiply(0.0001).rename("ndvi_lag")
    return lst_prev.addBands([qc_prev, ndvi_lag])


def era_window(t: ee.Date) -> ee.Image:
    """8-day weather summary ending the day before the target composite starts."""
    win = ERA.filterDate(t.advance(-8, "day"), t)
    sm = win.select("volumetric_soil_water_layer_1").mean().rename("soil_moisture")
    precip = win.select("total_precipitation_sum").sum().rename("precip_8d")
    dew = win.select("dewpoint_temperature_2m").mean().rename("dewpoint")
    rad = win.select("surface_solar_radiation_downwards_sum").sum().rename("solar_8d")
    # averaging/summing days loses ERA5's native grid, so restore it or points read the wrong pixel
    return sm.addBands([precip, dew, rad]).setDefaultProjection(ERA.first().projection())


def rows_modis(img: ee.Image, points: ee.FeatureCollection) -> ee.FeatureCollection:
    t = img.date()
    target = lst_celsius(img).rename("lst_target").addBands(img.select("QC_Day").rename("qc_target"))
    stack = target.addBands(sat_lags(t)).unmask(MISSING)
    rows = stack.sampleRegions(collection=points, scale=MODIS_SCALE)
    return rows.map(lambda f: f.set("date", t.format("YYYY-MM-dd")))


def rows_era(img: ee.Image, points: ee.FeatureCollection) -> ee.FeatureCollection:
    t = img.date()
    rows = era_window(t).unmask(MISSING).sampleRegions(collection=points, scale=ERA_SCALE)
    return rows.map(lambda f: f.set("date", t.format("YYYY-MM-dd")))
