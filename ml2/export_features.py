"""Export the modelling table, one CSV per summer, resumable.

    python -m ml2.export_features --years 2022          # one year
    python -m ml2.export_features                       # all 21 summers
    python -m ml2.export_features --static              # elevation + land cover only
"""
import argparse
import time

import ee
import pandas as pd

from ml2.config import EE_PROJECT, N_POINTS, RAW_DIR, SUMMER_END, SUMMER_START, YEARS

RETRIES = 4


def with_retry(fn, what: str):
    """Run an Earth Engine call, retrying transient failures; the last failure is raised, not hidden."""
    for attempt in range(1, RETRIES + 1):
        try:
            return fn()
        except ee.EEException as e:
            if attempt == RETRIES:
                raise
            wait = 5 * 2 ** attempt
            print(f"  {what}: {e} -- retry {attempt}/{RETRIES - 1} in {wait}s", flush=True)
            time.sleep(wait)


def composite_table(img, points) -> pd.DataFrame:
    from ml2.ee_features import rows_era, rows_modis
    from ml2.ee_points import rows_to_frame

    mod = with_retry(lambda: rows_to_frame(rows_modis(img, points)), "modis")
    era = with_retry(lambda: rows_to_frame(rows_era(img, points)), "era5")
    for name, df in (("modis", mod), ("era5", era)):
        if len(df) != N_POINTS:
            raise RuntimeError(f"{name} returned {len(df)} rows, expected {N_POINTS}")
    out = mod.merge(era.drop(columns=["lon", "lat"]), on=["pid", "date"], validate="one_to_one")
    out["doy"] = pd.to_datetime(out["date"]).dt.dayofyear
    return out


def export_year(year: int, points) -> None:
    from ml2.ee_features import LST_ALL

    path = RAW_DIR / f"features_{year}.csv"
    if path.exists():
        print(f"{year}: exists, skipping", flush=True)
        return
    col = LST_ALL.filterDate(f"{year}-{SUMMER_START}", f"{year}-{SUMMER_END}")
    starts = with_retry(lambda: col.aggregate_array("system:time_start").getInfo(), "list composites")
    frames = []
    for ms in starts:
        img = ee.Image(col.filterDate(ms, ms + 86_400_000).first())
        frames.append(composite_table(img, points))
        print(f"{year}: {pd.to_datetime(ms, unit='ms').date()} done", flush=True)
    table = pd.concat(frames, ignore_index=True)
    tmp = path.with_suffix(".tmp")
    table.to_csv(tmp, index=False)
    tmp.replace(path)                       # a half-written file never looks finished
    print(f"{year}: wrote {len(table)} rows", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--years", type=int, nargs="*", default=list(YEARS))
    ap.add_argument("--static", action="store_true", help="export only the static features")
    args = ap.parse_args()

    ee.Initialize(project=EE_PROJECT)       # must precede importing modules that build ee objects
    from ml2.ee_points import uk_points
    from ml2.ee_static import static_features

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    points = uk_points()

    static_path = RAW_DIR / "static_features.csv"
    if args.static or not static_path.exists():
        static = static_features(points)
        static.to_csv(static_path, index=False)
        print(f"static: wrote {static.shape}", flush=True)
    if args.static:
        return
    for year in args.years:
        export_year(year, points)


if __name__ == "__main__":
    main()
