"""Merge the per-year exports and static features into one table.

    python -m ml2.build_dataset
"""
import numpy as np
import pandas as pd

from ml2.config import RAW_DIR

PROCESSED = RAW_DIR.parent / "processed" / "dataset.parquet"


def load_raw() -> pd.DataFrame:
    files = sorted(RAW_DIR.glob("features_*.csv"))
    if not files:
        raise FileNotFoundError(f"no features_*.csv in {RAW_DIR}")
    frames = [pd.read_csv(f).assign(year=int(f.stem.split("_")[1])) for f in files]
    return pd.concat(frames, ignore_index=True)


def decode_qc(qc: pd.Series) -> pd.DataFrame:
    """MOD11A2 QC_Day bits: 0-1 mandatory flag, 6-7 LST error class (NaN stays NaN)."""
    q = qc.fillna(0).astype(np.int64).to_numpy()
    out = pd.DataFrame({"mand": q & 3, "err": (q >> 6) & 3}, index=qc.index)
    return out.where(qc.notna())


def always_missing_points(df: pd.DataFrame) -> list[int]:
    """Points whose LST target is never observed; they carry no information."""
    seen = df.groupby("pid")["lst_target"].apply(lambda s: s.notna().any())
    return sorted(seen[~seen].index.tolist())


def build() -> pd.DataFrame:
    df = load_raw()
    static = pd.read_csv(RAW_DIR / "static_features.csv")
    df = df.merge(static, on="pid", how="left", validate="many_to_one")

    q_t = decode_qc(df["qc_target"]).add_prefix("t_")
    q_p = decode_qc(df["qc_prev"]).add_prefix("p_")
    df = pd.concat([df, q_t, q_p], axis=1)

    dropped = always_missing_points(df)
    print(f"dropping {len(dropped)} always-missing points: {dropped}")
    df = df[~df["pid"].isin(dropped)].reset_index(drop=True)
    return df.sort_values(["pid", "date"]).reset_index(drop=True)


def main() -> None:
    df = build()
    PROCESSED.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(PROCESSED, index=False)
    print(f"wrote {PROCESSED}: {df.shape}")


if __name__ == "__main__":
    main()
