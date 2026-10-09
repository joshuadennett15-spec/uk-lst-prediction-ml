"""Data checks on the raw exports. Prints a report; exits non-zero if a hard check fails.

    python -m ml2.validate
"""
import sys

import numpy as np
import pandas as pd

from ml2.build_dataset import always_missing_points, load_raw
from ml2.config import N_POINTS, YEARS

# plausible UK summer ranges; counted, not enforced, because a few cold-bias pixels are expected
RANGES = {"lst_target": (0, 50), "lst_prev": (0, 50), "ndvi_lag": (-0.2, 1.0),
          "soil_moisture": (0, 0.8), "dewpoint": (265, 295)}

failures: list[str] = []


def hard(name: str, ok: bool, detail: str = "") -> None:
    print(f"[{'PASS' if ok else 'FAIL'}] {name} {detail}")
    if not ok:
        failures.append(name)


def check_integrity(df: pd.DataFrame) -> None:
    print("\n== 1. Integrity ==")
    hard("all 21 years present", sorted(df["year"].unique()) == list(YEARS))
    per_date = df.groupby(["year", "date"])["pid"].nunique()
    hard(f"every date has {N_POINTS} points", (per_date == N_POINTS).all(), f"(min {per_date.min()})")
    hard("no duplicate (pid, date)", not df.duplicated(["pid", "date"]).any())
    n_comp = df.groupby("year")["date"].nunique()
    print(f"     composites per year: {n_comp.min()}-{n_comp.max()}")


def check_plausibility(df: pd.DataFrame) -> None:
    print("\n== 2. Plausibility (values outside summer UK ranges) ==")
    for col, (lo, hi) in RANGES.items():
        bad = df[col].notna() & ((df[col] < lo) | (df[col] > hi))
        print(f"     {col}: {int(bad.sum())} out of range [{lo}, {hi}]")
    yearly = df.groupby("year")["lst_target"].mean().round(1)
    print(f"     yearly mean LST: min {yearly.min()} ({yearly.idxmin()}), max {yearly.max()} ({yearly.idxmax()})")


def check_lag_alignment(df: pd.DataFrame) -> None:
    print("\n== 3. Lag alignment (leakage) ==")
    d = df[["pid", "year", "date", "lst_target", "lst_prev"]].copy()
    d["date"] = pd.to_datetime(d["date"])
    dates = d[["year", "date"]].drop_duplicates().sort_values(["year", "date"])
    dates["prev_date"] = dates.groupby("year")["date"].shift(1)
    gap = (dates["date"] - dates["prev_date"]).dropna().dt.days
    hard("consecutive composites are 8 days apart", (gap == 8).all(), f"(gaps seen: {sorted(gap.unique())})")

    d = d.merge(dates, on=["year", "date"]).dropna(subset=["prev_date"])
    earlier = d[["pid", "date", "lst_target"]].rename(columns={"date": "prev_date", "lst_target": "earlier_target"})
    m = d.merge(earlier, on=["pid", "prev_date"], validate="one_to_one")
    same_nan = (m["lst_prev"].isna() == m["earlier_target"].isna()).mean()
    diff = (m["lst_prev"] - m["earlier_target"]).abs().max()
    hard("lst_prev equals the previous composite's lst_target", diff < 1e-6, f"(max diff {diff:.2g})")
    hard("missing pattern identical", same_nan == 1.0, f"({same_nan:.4f})")


def check_missingness(df: pd.DataFrame) -> None:
    print("\n== 4. Missingness ==")
    miss = df["lst_target"].isna()
    print("     LST missing by year:", df.assign(m=miss).groupby("year")["m"].mean().round(2).to_dict())
    worst = df.assign(m=miss).groupby(["year", "date"])["m"].mean().nlargest(3).round(2)
    print("     cloudiest composites:", {f"{y} {d}": v for (y, d), v in worst.items()})

    gone = always_missing_points(df)
    era_missing = df.groupby("pid")["soil_moisture"].apply(lambda s: s.isna().all())
    era_pts = set(era_missing[era_missing].index)
    print(f"     points never observed in LST: {len(gone)}; with no ERA5 at all: {len(era_pts)}; "
          f"overlap: {len(era_pts & set(gone))}")

    pm = df.assign(m=miss).groupby("pid")["m"].mean()
    pts = df.drop_duplicates("pid").set_index("pid")[["lat", "lon"]]
    corr_lat = pm.corr(pts["lat"])
    print(f"     per-point missing rate vs latitude: r = {corr_lat:.2f}")
    warm_prev = df["lst_prev"] > df["lst_prev"].quantile(0.9)
    print(f"     LST missing rate overall {miss.mean():.3f}; when previous LST was in its hottest decile "
          f"{miss[warm_prev].mean():.3f}")


def main() -> int:
    df = load_raw()
    print(f"loaded {df.shape[0]} rows, {df.shape[1]} columns")
    check_integrity(df)
    check_plausibility(df)
    check_lag_alignment(df)
    check_missingness(df)
    print("\n" + ("ALL HARD CHECKS PASSED" if not failures else f"FAILED: {failures}"))
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
