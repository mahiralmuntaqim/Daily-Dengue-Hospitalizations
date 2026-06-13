# Forecasting Daily Dengue Hospitalizations in Dhaka Using Weather Data

A machine learning study comparing tree-based ensemble models for predicting daily dengue hospitalizations in Dhaka, Bangladesh, using meteorological features and temporal lag engineering.

> **Paper:** *Forecasting Daily Dengue Hospitalizations in Dhaka using weather data*
> **Authors:** Mahir Al Muntaqim · Ahnaf Hossain Rauf
> **Affiliation:** Department of Computer Science and Engineering, BRAC University, Dhaka, Bangladesh

---

## Overview

Dengue fever has become a near-permanent pressure on Dhaka's urban health system — Bangladesh recorded its worst-ever outbreak in 2023 with 321,179 confirmed cases and 1,705 deaths. This project builds a **daily-resolution early warning system** to predict new dengue hospitalizations, giving hospitals actionable short-horizon forecasts for bed allocation, IV-fluid stock, platelet supply, and staff planning.

Four tree-based regressors are trained on a unified daily dataset (April 2021 – December 2025) with identical preprocessing and chronological splits, enabling a clean apples-to-apples comparison.

---

## Models Compared

| Model | Key Configuration |
|---|---|
| **Random Forest** | `n_estimators=110`, `max_depth=None`, `min_samples_leaf=3` |
| **Gradient Boosting** | `n_estimators=110`, `lr=0.08`, `max_depth=4`, `subsample=0.8` |
| **XGBoost** | `n_estimators=100`, `lr=0.08`, `max_depth=5` |
| **LightGBM** | `n_estimators=150`, `lr=0.05`, `max_depth=5`, `reg_alpha=0.1` |

---

## Dataset

| Property | Value |
|---|---|
| City | Dhaka, Bangladesh |
| Date range | 10 Apr 2021 – 31 Dec 2025 |
| Total rows (modeling) | 1,706 |
| Features | 31 |
| Target | `log1p(new_hospitalizations)` |
| Train / Test split | 80 / 20 chronological |
| Train rows | 1,364 (up to 23 Jan 2025) |
| Test rows | 342 (24 Jan – 31 Dec 2025) |

**Sources:** DGHS dengue surveillance data · Bangladesh Meteorological Department · NASA

### Features

- **Meteorological:** daily mean temperature, temperature range, humidity, dew point, precipitation, wind speed, atmospheric pressure, solar radiation, sunshine hours
- **Climate lags:** 7-, 14-, and 21-day lagged precipitation and humidity (matched to Aedes aegypti breeding and incubation windows)
- **Target lags:** previous-day hospitalizations, 7-day-ago hospitalizations, shifted 7-day moving average
- **Categorical:** season (pre-monsoon / monsoon / post-monsoon), dominant DENV serotype, ordinal alert level
- **Cyclical encoding:** month encoded as sine/cosine pair to avoid December–January discontinuity

---

## Results

### Test-Set Performance (Jan – Dec 2025)

| Model | MAE | RMSE | R² | CV Mean R² | CV Std R² |
|---|---|---|---|---|---|
| Random Forest | 32.29 | 68.67 | 0.7728 | 0.8764 | 0.0631 |
| **Gradient Boosting** | 31.52 | **65.67** | **0.7922** | **0.9255** | **0.0196** |
| XGBoost | **31.50** | 67.15 | 0.7828 | 0.9226 | 0.0276 |
| LightGBM | 31.85 | 67.77 | 0.7787 | 0.9149 | 0.0384 |

*All metrics in log space. CV = 5-fold TimeSeriesSplit cross-validation.*

### Outbreak Classification (ROC AUC)

Threshold: ≥ 234 hospitalizations/day = outbreak day

| Model | AUC |
|---|---|
| Gradient Boosting | **0.986** |
| LightGBM | 0.985 |
| Random Forest | 0.981 |
| XGBoost | 0.979 |

### Key Findings

- **Boosting methods outperform Random Forest on stability** — Gradient Boosting achieves CV R² of 0.9255 ± 0.0196 vs. Random Forest's 0.8764 ± 0.0631
- **Model family matters less than feature engineering** at this scale (~1,300 training rows); XGBoost, LightGBM, and Gradient Boosting are within 0.005 of each other on test R²
- **Weather data adds causal grounding** — even though adding meteorological variables slightly reduced raw metrics (likely because short-term target lags already carry strong signal), weather features provide mechanistic links essential for an early warning system
- **Alert level leakage flag** — `alert_level_enc` dominates feature importance in boosting models; if the alert level was assigned using same-day hospitalization counts, this creates a near-leakage situation that warrants a controlled rerun

---

## Preprocessing Pipeline

1. **Ordinal encoding** of alert level (`No Alert=0` → `Critical=7`)
2. **One-hot encoding** of season and DENV serotype (first level dropped)
3. **Cyclical month encoding** (sine/cosine on a 12-month period)
4. **Log1p transformation** of target to stabilize high variance (raw mean: 157.2, std: 217.1, max: 1,771)
5. **Target lag construction** with a one-day shift on the 7-day moving average to prevent leakage
6. Dropped first 21 rows to remove NaNs from lag construction

---

## Limitations

- **Single city** — Dhaka only; spatial transferability to other Bangladeshi divisions is untested
- **Short time span** — 4.5 years may not capture full serotype-replacement cycles
- **Alert level leakage** — `alert_level_enc` feature may encode same-day hospitalization information
- **No hyperparameter tuning** — reasonable defaults used; grid search / Bayesian optimization may shift model rankings
- **No external baseline** — no persistence, seasonal-naive, or ARIMA comparison included

---

## Future Work

- [ ] Rerun pipeline without `alert_level_enc` (or with a strictly lagged version) to isolate the climate-driven signal
- [ ] Add seasonal-naive and SARIMA baselines for fair external comparison
- [ ] Train LSTM-attention model on the same daily series
- [ ] Extend geographic scope to all eight Bangladeshi divisions
- [ ] Integrate SHAP-based interpretability for public-health stakeholders

---

## Tech Stack

| Tool | Purpose |
|---|---|
| `pandas` / `NumPy` | Data handling |
| `scikit-learn` | Random Forest, Gradient Boosting |
| `xgboost` | XGBoost |
| `lightgbm` | LightGBM |
| `matplotlib` / `seaborn` | Visualization |

All experiments run with `random_state=42`.

---

## Citation

If you use this work, please cite:

```
Mahir Al Muntaqim and Ahnaf Hossain Rauf, "Forecasting Daily Dengue Hospitalizations 
in Dhaka using weather data," Department of Computer Science and Engineering, 
BRAC University, Dhaka, Bangladesh.
```

---

## References

Key references from the paper:

- Dey et al. (2022) — DengueBD dataset, SVR baseline, PLOS ONE
- Rahman et al. (2025) — Dengue early warning system, Bangladesh, Health Sci. Rep.
- Liu et al. (2025) — Multi-model comparison across Bangladeshi divisions, Sci. Rep.
- Al Mobin (2024) — Stepwise feature selection for dengue forecasting, Sci. Rep.
- Hossain et al. (2023) — 22-year epidemiological retrospective, Trop. Med. Health
- Chen & Guestrin (2016) — XGBoost, KDD
- Ke et al. (2017) — LightGBM, NeurIPS
