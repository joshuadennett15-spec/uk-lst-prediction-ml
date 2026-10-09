# UK land surface temperature prediction

Can weather, vegetation, elevation and land cover predict the next 8-day land surface temperature (LST) at a UK location better than a simple seasonal average?

**Short answer: yes, but only by about 3.5%.** Ridge regression and LightGBM both beat the best baseline, and they finish within 0.01 °C of each other.

## Data

| | |
|---|---|
| Target | MODIS 8-day LST for the next composite, in °C |
| Points | 1,984 locations sampled across the UK |
| Period | June to August, 2003-2023 (21 summers, 252 composites per point) |
| Predictors | previous composite's LST, lagged NDVI, ERA5-Land dewpoint, 8-day precipitation, soil moisture and solar radiation, day of year, latitude, longitude, elevation, land-cover fractions |
| Size | 499,968 rows; 17.9% have no LST (cloud) and are dropped, leaving 410,417 |

Everything is pulled from Google Earth Engine by `ml2/export_features.py` and assembled by `ml2/build_dataset.py`.

## Method

**1. Check for leakage before splitting** ([notebook 02](notebooks/02_checks.ipynb)). Nearby points have correlated temperature anomalies: 0.79 within 10 km, falling to 0.25 beyond 800 km, never reaching zero. A random split would put near-copies of the same weather in both train and test and flatter the score. So the split is by time: **train on 2003-2018, test on 2019-2023.**

![Correlation of LST anomalies against distance](figures/01_distance_decay.png)

**2. Set the bar with baselines** ([notebook 03](notebooks/03_baselines.ipynb)). Any model has to beat guesses that need no learning. Scores are test RMSE on the 76,310 test rows that have a previous composite.

**3. Fit models** ([notebook 04](notebooks/04_models.ipynb)). Ridge regression (median imputation, standardised) and LightGBM (329 trees, chosen by early stopping on 2016-2018 held out from the training years).

## Results

| Model | Test RMSE (°C) |
|---|---|
| Spread of the data (always guess the mean) | 5.06 |
| Persistence (repeat the last composite) | 5.04 |
| Climatology (per-point, per-composite training mean) | 4.44 |
| Climatology + 0.29 × last departure from climatology | 4.21 |
| **Ridge regression** | **4.07** |
| **LightGBM** | **4.06** |

![Test RMSE for each baseline and model](figures/02_rmse_ladder.png)

What this says:

- **Place and season carry most of the signal.** Climatology alone cuts RMSE from 5.06 to 4.44. Repeating the last composite (5.04) is barely better than guessing the mean, because a single composite is noisy.
- **The models add a little on top.** Against the best baseline (4.21), Ridge and LightGBM each reduce RMSE by about 0.15 °C, roughly 3.5%.
- **Trees add nothing over a linear model here.** Ridge and LightGBM are within 0.01 °C, so with these features the non-linearity isn't being used.
- **LightGBM fits the training years more tightly than the test years** (3.30 train vs 4.06 test; Ridge is 3.75 vs 4.07), so it overfits a little without gaining accuracy on unseen years.

![Test RMSE by year](figures/03_rmse_by_year.png)

![LightGBM predicted against actual LST](figures/04_predicted_vs_actual.png)

![LightGBM feature importance](figures/05_feature_importance.png)

## Limitations

- **Unseen places are not tested.** The split holds out years, not locations, so these scores say how well the model forecasts known places in later years, not how it would do somewhere new.
- **Cloud gaps.** About 18% of LST values are missing and dropped, and 17% of rows have no previous composite. Scores are only on rows where a previous composite exists, so they don't cover cases where the previous composite was cloud-covered.
- **Large irreducible error.** An RMSE near 4 °C on 8-day composites means most of the variation is not explained by these features.

## Reproduce

```bash
pip install -r requirements.txt            # file is UTF-16; convert it if your pip complains
earthengine authenticate
export EE_PROJECT=<your-earth-engine-project>
python -m ml2.export_features              # exports from Earth Engine
python -m ml2.build_dataset                # writes data/processed/dataset.parquet
python -m ml2.validate                     # data checks
python scripts/make_figures.py             # rebuilds the figures above
```

Then run notebooks `02_checks`, `03_baselines` and `04_models` in order. Raw and processed data are not committed (`data/` is git-ignored); they are rebuilt by the steps above.
