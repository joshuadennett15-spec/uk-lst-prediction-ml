"""Rebuild the README figures from data/processed/dataset.parquet.

Mirrors notebooks 02-04: same split (train 2003-2018, test 2019-2023), same
baselines, same Ridge and LightGBM settings. Run from the repo root:
    python scripts/make_figures.py
"""
from pathlib import Path

import lightgbm as lgb
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

DATA = Path("data/processed/dataset.parquet")
OUT = Path("figures")

SURFACE, INK, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e3e2dd"
BLUE, ORANGE, GREY = "#2a78d6", "#eb6834", "#a8a79f"

FEATURES = ["lst_prev", "ndvi_lag", "dewpoint", "precip_8d", "soil_moisture", "solar_8d",
            "doy", "lat", "lon", "elevation",
            "lc10", "lc20", "lc30", "lc40", "lc50", "lc60", "lc70", "lc80", "lc90", "lc95", "lc100",
            "p_mand", "p_err"]
TARGET = "lst_target"


def rmse(y, p):
    return float(np.sqrt(np.mean((np.asarray(y) - np.asarray(p)) ** 2)))


def style(ax):
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.xaxis.label.set_color(MUTED)
    ax.yaxis.label.set_color(MUTED)
    ax.title.set_color(INK)
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.set_axisbelow(True)


def new_fig(*args, **kw):
    fig, ax = plt.subplots(*args, **kw)
    fig.patch.set_facecolor(SURFACE)
    for a in np.atleast_1d(ax).ravel():
        style(a)
    return fig, ax


def distance_decay(df):
    """Fig 1: correlation of LST anomalies vs distance between points (notebook 02)."""
    d = df.dropna(subset=[TARGET]).copy()
    d["anom"] = d[TARGET] - d.groupby(["pid", "comp"])[TARGET].transform("mean")
    corr = d.pivot(index="date", columns="pid", values="anom").corr(min_periods=100)
    pts = d.drop_duplicates("pid").set_index("pid")[["lat", "lon"]].loc[corr.columns]
    lat, lon = np.radians(pts["lat"].to_numpy()), np.radians(pts["lon"].to_numpy())
    h = (np.sin((lat[:, None] - lat[None, :]) / 2) ** 2
         + np.cos(lat[:, None]) * np.cos(lat[None, :]) * np.sin((lon[:, None] - lon[None, :]) / 2) ** 2)
    dist = 2 * 6371 * np.arcsin(np.sqrt(h))
    iu = np.triu_indices(len(dist), k=1)
    pairs = pd.DataFrame({"km": dist[iu], "r": corr.to_numpy()[iu]}).dropna()
    bins = [0, 10, 25, 50, 100, 200, 400, 800, 1300]
    pairs["band"] = pd.cut(pairs["km"], bins)
    g = pairs.groupby("band", observed=True)["r"].mean()

    fig, ax = new_fig(figsize=(7, 4))
    x = np.arange(len(g))
    ax.bar(x, g.values, color=BLUE, width=0.62)
    for xi, v in zip(x, g.values):
        ax.text(xi, v + 0.015, f"{v:.2f}", ha="center", fontsize=9, color=INK)
    ax.set_xticks(x, [f"{int(b.left)}-{int(b.right)}" for b in g.index])
    ax.set_xlabel("distance between two points (km)")
    ax.set_ylabel("mean correlation of LST anomalies")
    ax.set_ylim(0, 0.9)
    ax.set_title("Nearby points move together, so the split has to be by time", loc="left", fontsize=11)
    fig.tight_layout()
    fig.savefig(OUT / "01_distance_decay.png", dpi=160)
    plt.close(fig)


def main():
    OUT.mkdir(exist_ok=True)
    df = pd.read_parquet(DATA)
    df["comp"] = df.groupby(["pid", "year"]).cumcount()
    distance_decay(df)

    df = df.dropna(subset=[TARGET])
    train, test = df[df["year"] <= 2018], df[df["year"] >= 2019]
    X_train, y_train, X_test, y_test = train[FEATURES], train[TARGET], test[FEATURES], test[TARGET]
    same_rows = ((test["comp"] > 0) & test["lst_prev"].notna()).to_numpy()

    # baselines (notebook 03), evaluated on the same rows as the models
    clim = train.groupby(["pid", "comp"])[TARGET].mean().rename("clim")
    prev_clim = clim.rename("prev_clim").reset_index()
    prev_clim["comp"] += 1

    def prep(d):
        d = d.join(clim, on=["pid", "comp"]).merge(prev_clim, on=["pid", "comp"], how="left")
        d["prev_anom"] = d["lst_prev"] - d["prev_clim"]
        d["anom"] = d[TARGET] - d["clim"]
        return d.dropna(subset=["clim", "prev_anom"])

    tr, te = prep(train), prep(test)
    k = (tr["prev_anom"] * tr["anom"]).sum() / (tr["prev_anom"] ** 2).sum()
    spread = float(te[TARGET].std(ddof=0))
    base = {
        "Spread (guess the mean)": spread,
        "Persistence": rmse(te[TARGET], te["lst_prev"]),
        "Climatology": rmse(te[TARGET], te["clim"]),
        f"Climatology + {k:.2f} x last departure": rmse(te[TARGET], te["clim"] + k * te["prev_anom"]),
    }

    ridge = make_pipeline(SimpleImputer(strategy="median", add_indicator=True),
                          StandardScaler(), Ridge(alpha=1.0)).fit(X_train, y_train)
    pred_ridge = ridge.predict(X_test)

    params = dict(learning_rate=0.05, num_leaves=15, min_child_samples=20, subsample=0.8,
                  subsample_freq=1, colsample_bytree=0.7, random_state=42, verbose=-1)
    fit, val = train[train["year"] <= 2015], train[train["year"] >= 2016]
    probe = lgb.LGBMRegressor(n_estimators=3000, **params).fit(
        fit[FEATURES], fit[TARGET], eval_set=[(val[FEATURES], val[TARGET])],
        eval_metric="rmse", callbacks=[lgb.early_stopping(50, verbose=False)])
    lgbm = lgb.LGBMRegressor(n_estimators=probe.best_iteration_, **params).fit(X_train, y_train)
    pred_lgbm = lgbm.predict(X_test)

    actual = y_test.to_numpy()[same_rows]
    r_ridge, r_lgbm = rmse(actual, pred_ridge[same_rows]), rmse(actual, pred_lgbm[same_rows])
    best = list(base.values())[-1]
    print(f"rows scored: {same_rows.sum()}  trees: {probe.best_iteration_}  k: {k:.3f}")
    for name, v in base.items():
        print(f"{name:40s} {v:.3f}")
    print(f"{'Ridge':40s} {r_ridge:.3f}   LightGBM {r_lgbm:.3f}")
    print(f"gain vs best baseline: ridge {100*(best-r_ridge)/best:.1f}%  lgbm {100*(best-r_lgbm)/best:.1f}%")
    print(f"R2 vs spread: ridge {1-(r_ridge/spread)**2:.3f}  lgbm {1-(r_lgbm/spread)**2:.3f}  best baseline {1-(best/spread)**2:.3f}")

    # Fig 2: RMSE ladder
    labels = list(base) + ["Ridge", "LightGBM"]
    vals = list(base.values()) + [r_ridge, r_lgbm]
    colors = [GREY] * len(base) + [BLUE, ORANGE]
    fig, ax = new_fig(figsize=(8, 4))
    ax.barh(labels[::-1], vals[::-1], color=colors[::-1], height=0.6)
    for y, v in enumerate(vals[::-1]):
        ax.text(v + 0.04, y, f"{v:.2f}", va="center", fontsize=9, color=INK)
    ax.set_xlim(3.5, 5.4)
    ax.grid(axis="y", visible=False)
    ax.grid(axis="x", color=GRID, lw=0.8)
    ax.set_xlabel("test RMSE, 2019-2023 (deg C)  -  axis starts at 3.5")
    ax.set_title("The models beat the best baseline, but only modestly", loc="left", fontsize=11)
    fig.tight_layout()
    fig.savefig(OUT / "02_rmse_ladder.png", dpi=160)
    plt.close(fig)

    # Fig 3: RMSE by year
    rows = test[same_rows].assign(ridge=pred_ridge[same_rows], lgbm=pred_lgbm[same_rows])
    te_b = te.assign(base=te["clim"] + k * te["prev_anom"])
    years = sorted(rows["year"].unique())
    yr = lambda d, c: [rmse(d.loc[d["year"] == y, TARGET], d.loc[d["year"] == y, c]) for y in years]
    x, w = np.arange(len(years)), 0.27
    fig, ax = new_fig(figsize=(8, 4))
    ax.bar(x - w, yr(te_b, "base"), w, color=GREY, label="Best baseline")
    ax.bar(x, yr(rows, "ridge"), w, color=BLUE, label="Ridge")
    ax.bar(x + w, yr(rows, "lgbm"), w, color=ORANGE, label="LightGBM")
    ax.set_xticks(x, years)
    ax.set_ylabel("test RMSE (deg C)")
    ax.set_title("Per-year test RMSE, same rows for every model", loc="left", fontsize=11)
    ax.legend(frameon=False, fontsize=9, labelcolor=MUTED, ncol=3, loc="upper center",
              bbox_to_anchor=(0.5, -0.12))
    fig.tight_layout()
    fig.savefig(OUT / "03_rmse_by_year.png", dpi=160)
    plt.close(fig)
    print("by year  base:", np.round(yr(te_b, "base"), 2), " ridge:", np.round(yr(rows, "ridge"), 2),
          " lgbm:", np.round(yr(rows, "lgbm"), 2))

    # Fig 4: predicted vs actual (LightGBM)
    lim = (-10, 50)
    fig, ax = new_fig(figsize=(5.6, 5))
    ax.grid(visible=False)
    hb = ax.hexbin(actual, pred_lgbm[same_rows], gridsize=60, extent=(*lim, *lim), mincnt=1,
                   bins="log", cmap="Blues")
    ax.plot(lim, lim, color=ORANGE, lw=1.2)
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ax.set_xlabel("actual LST (deg C)")
    ax.set_ylabel("predicted LST (deg C)")
    ax.set_title("LightGBM: predicted vs actual", loc="left", fontsize=11)
    cb = fig.colorbar(hb, ax=ax)
    cb.set_label("rows (log scale)", color=MUTED)
    cb.ax.tick_params(colors=MUTED, labelsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "04_predicted_vs_actual.png", dpi=160)
    plt.close(fig)

    # Fig 5: feature importance
    imp = pd.Series(lgbm.booster_.feature_importance(importance_type="gain"), index=FEATURES)
    imp = (imp / imp.sum()).sort_values()
    print("top importance:", imp.sort_values(ascending=False).head(8).round(3).to_dict())
    top = imp.tail(10)
    fig, ax = new_fig(figsize=(6.5, 4))
    ax.barh(top.index, top.values, color=BLUE, height=0.6)
    for y, v in enumerate(top.values):
        ax.text(v + 0.004, y, f"{v:.0%}", va="center", fontsize=9, color=INK)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("share of total gain")
    ax.set_title("What LightGBM leans on (top 10 of 23 features)", loc="left", fontsize=11)
    fig.tight_layout()
    fig.savefig(OUT / "05_feature_importance.png", dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    main()
