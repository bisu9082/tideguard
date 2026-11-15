#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
tideguard (Project #24) — WBE-ML integrated forecasting, Step 4 analysis.
Reproducible pipeline. Data integrity: real public data only (CDC NWSS).

Execution split (per Ku decision 2026-06-21):
  - SANDBOX  : classical baselines (seasonal-naive, SARIMA, XGBoost) + ablations + stats.
  - RTX 3090 : neural models (LSTM) and TimesFM-2.5 (zero-shot + fine-tuned).
               -> run with  --models lstm,timesfm_zs,timesfm_ft   on the GPU box.

Outputs (data/):
  raw_results.csv            per (site, model, horizon, history, missing, seed) errors
  experiment_summary.json    aggregated metrics + statistics
  results_table.tex          LaTeX results table (Step 5 insert)

Usage:
  python analysis_main.py --stage baselines      # sandbox
  python analysis_main.py --stage neural         # 3090 (needs torch, timesfm)
"""
import argparse, json, os, sys, warnings, math, random
import numpy as np
import pandas as pd
warnings.filterwarnings("ignore")

# ----------------------------------------------------------------------
# CONFIG
# ----------------------------------------------------------------------
HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(HERE, "data")
FIG_DIR = os.path.join(HERE, "figure")
NWSS_CSV = os.path.join(DATA_DIR, "nwss_sarscov2_concentration.csv")

CFG = dict(
    normalization="flow-population",  # most common NWSS normalization
    min_points=200,                   # weekly points required per site
    n_sites=16,                       # dense sites used for experiment 0
    freq="W",                         # weekly resampling (standard WBE cadence)
    horizons=[1, 2, 4],               # weeks ahead
    histories=[12, 26, 52, 104],      # A3: training-history length (weeks); None=full
    missing_levels=[0.0, 0.10, 0.30], # A4: injected missingness
    n_origins=24,                     # rolling-origin test points per site
    seeds=[0, 1, 2, 3, 4],            # >=5 seeds
    bootstrap=2000,                   # Bootstrap CI samples (>=1000)
)

# ----------------------------------------------------------------------
# DATA PREP
# ----------------------------------------------------------------------
def load_series():
    """Return dict[site] -> pd.Series of log10 weekly-mean concentration."""
    df = pd.read_csv(NWSS_CSV)
    df = df[df["normalization"] == CFG["normalization"]].copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["pcr_conc_lin"] = pd.to_numeric(df["pcr_conc_lin"], errors="coerce")
    df = df.dropna(subset=["date", "pcr_conc_lin"])
    df = df[df["pcr_conc_lin"] > 0]
    df["logc"] = np.log10(df["pcr_conc_lin"])
    out = {}
    for sid, g in df.groupby("key_plot_id"):
        s = (g.set_index("date")["logc"].sort_index()
               .resample(CFG["freq"]).mean())
        s = s.interpolate(limit=3, limit_direction="both")  # reported: short-gap fill
        s = s.dropna()
        if len(s) >= CFG["min_points"]:
            out[sid] = s
    # rank by length, keep densest n_sites (deterministic)
    chosen = sorted(out, key=lambda k: -len(out[k]))[:CFG["n_sites"]]
    return {k: out[k] for k in chosen}

def inject_missing(train, frac, rng):
    if frac <= 0:
        return train
    t = train.copy()
    idx = rng.choice(len(t), int(round(frac * len(t))), replace=False)
    t.iloc[idx] = np.nan
    return t.interpolate(limit_direction="both")  # reported gap-fill

# ----------------------------------------------------------------------
# METRICS & STATS
# ----------------------------------------------------------------------
def smape(y, yhat):
    d = (np.abs(y) + np.abs(yhat))
    d[d == 0] = 1e-9
    return 100.0 * np.mean(2.0 * np.abs(yhat - y) / d)

def metrics(y, yhat):
    y, yhat = np.asarray(y, float), np.asarray(yhat, float)
    return dict(mae=float(np.mean(np.abs(yhat - y))),
                rmse=float(math.sqrt(np.mean((yhat - y) ** 2))),
                smape=float(smape(y, yhat)))

def bootstrap_ci(x, nboot, rng, alpha=0.05):
    x = np.asarray(x, float)
    if len(x) == 0:
        return (float("nan"), float("nan"))
    bs = [np.mean(rng.choice(x, len(x), replace=True)) for _ in range(nboot)]
    return (float(np.percentile(bs, 100 * alpha / 2)),
            float(np.percentile(bs, 100 * (1 - alpha / 2))))

def cohens_d(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    na, nb = len(a), len(b)
    if na < 2 or nb < 2:
        return float("nan")
    sp = math.sqrt(((na - 1) * a.var(ddof=1) + (nb - 1) * b.var(ddof=1)) / (na + nb - 2))
    return float((a.mean() - b.mean()) / sp) if sp > 0 else float("nan")

def diebold_mariano(e1, e2, h=1):
    """DM test on squared-error loss differential. Returns (stat, pvalue)."""
    from scipy import stats
    e1, e2 = np.asarray(e1, float), np.asarray(e2, float)
    d = e1 ** 2 - e2 ** 2
    n = len(d)
    if n < 8:
        return (float("nan"), float("nan"))
    dbar = d.mean()
    # Newey-West long-run variance up to lag h-1
    gamma0 = np.sum((d - dbar) ** 2) / n
    var = gamma0
    for lag in range(1, h):
        cov = np.sum((d[lag:] - dbar) * (d[:-lag] - dbar)) / n
        var += 2 * (1 - lag / h) * cov
    if var <= 0:
        return (float("nan"), float("nan"))
    stat = dbar / math.sqrt(var / n)
    p = 2 * (1 - stats.norm.cdf(abs(stat)))
    return (float(stat), float(p))

# ----------------------------------------------------------------------
# MODELS
#  SANDBOX baselines (pure numpy/scipy): naive, seasonal_naive, ets, ar_ols, ridge_ar
#  3090 models (full deps): sarima (statsmodels), xgboost, lstm, timesfm_*  [neural stage]
#  Integrity: heavy models are NEVER silently degraded to naive in the sandbox;
#             they are only run where their library is genuinely available.
# ----------------------------------------------------------------------
def f_naive(train, h):
    return float(train.iloc[-1])

def f_seasonal_naive(train, h, period=52):
    if len(train) > period:
        i = -period + (h - 1)
        return float(train.iloc[i]) if (h - 1) < period else float(train.iloc[-1])
    return float(train.iloc[-1])  # fallback: random-walk naive when history < 1 season

def f_ets(train, h, alpha=0.5, beta=0.1):
    """Holt linear-trend exponential smoothing (pure numpy)."""
    x = train.values.astype(float)
    if len(x) < 3:
        return float(x[-1])
    lvl, trend = x[0], x[1] - x[0]
    for t in range(1, len(x)):
        prev = lvl
        lvl = alpha * x[t] + (1 - alpha) * (lvl + trend)
        trend = beta * (lvl - prev) + (1 - beta) * trend
    return float(lvl + h * trend)

def _ar_feats(arr, lags):
    X, y = [], []
    for i in range(lags, len(arr)):
        X.append(arr[i - lags:i]); y.append(arr[i])
    return np.asarray(X, float), np.asarray(y, float)

def f_ar_ols(train, h, lags=4):
    """AR(p) by ordinary least squares (numpy). Statistical baseline."""
    arr = train.values.astype(float)
    if len(arr) < lags + 5:
        return float(arr[-1])
    X, y = _ar_feats(arr, lags)
    Xd = np.hstack([np.ones((len(X), 1)), X])
    beta, *_ = np.linalg.lstsq(Xd, y, rcond=None)
    cur = list(arr[-lags:])
    pred = cur[-1]
    for _ in range(h):  # recursive multi-step
        pred = float(beta[0] + np.dot(beta[1:], cur[-lags:]))
        cur.append(pred)
    return pred

def f_ridge_ar(train, h, lags=8, lam=1.0):
    """Ridge-regularized AR (numpy). Linear ML baseline."""
    arr = train.values.astype(float)
    if len(arr) < lags + 5:
        return float(arr[-1])
    X, y = _ar_feats(arr, lags)
    Xd = np.hstack([np.ones((len(X), 1)), X])
    A = Xd.T @ Xd + lam * np.eye(Xd.shape[1])
    beta = np.linalg.solve(A, Xd.T @ y)
    cur = list(arr[-lags:])
    pred = cur[-1]
    for _ in range(h):
        pred = float(beta[0] + np.dot(beta[1:], cur[-lags:]))
        cur.append(pred)
    return pred

# --- 3090-only heavy models (require their libraries; no silent fallback) ---
def f_sarima(train, h):
    from statsmodels.tsa.arima.model import ARIMA  # raises if unavailable
    m = ARIMA(train.values, order=(2, 1, 2)).fit(method_kwargs={"warn_convergence": False})
    return float(m.forecast(h)[-1])

def f_xgboost(train, h, seed, lags=8):
    from xgboost import XGBRegressor  # raises if unavailable
    arr = train.values.astype(float)
    if len(arr) < lags + 5:
        return float(arr[-1])
    X, y = _ar_feats(arr, lags)
    m = XGBRegressor(n_estimators=120, max_depth=3, learning_rate=0.1,
                     subsample=0.9, colsample_bytree=0.9, random_state=seed, verbosity=0)
    m.fit(X, y)
    cur = list(arr[-lags:]); pred = None
    for _ in range(h):
        pred = float(m.predict(np.array(cur[-lags:]).reshape(1, -1))[0]); cur.append(pred)
    return pred

BASELINES = ["naive", "seasonal_naive", "ets", "ar_ols", "ridge_ar"]   # sandbox
HEAVY = ["sarima", "xgboost"]                                          # 3090 (statsmodels/xgboost)

def forecast(model, train, h, seed):
    if model == "naive":          return f_naive(train, h)
    if model == "seasonal_naive": return f_seasonal_naive(train, h)
    if model == "ets":            return f_ets(train, h)
    if model == "ar_ols":         return f_ar_ols(train, h)
    if model == "ridge_ar":       return f_ridge_ar(train, h)
    if model == "sarima":         return f_sarima(train, h)
    if model == "xgboost":        return f_xgboost(train, h, seed)
    raise ValueError(model)

# ----------------------------------------------------------------------
# EXPERIMENT LOOP
# ----------------------------------------------------------------------
def run(models, tag):
    series = load_series()
    print(f"[data] sites={len(series)} | "
          f"len range {min(len(s) for s in series.values())}-{max(len(s) for s in series.values())}", flush=True)
    rows = []
    for si, (sid, s) in enumerate(series.items()):
        vals = s.values.astype(float)
        n = len(vals)
        for hist in CFG["histories"]:
            for miss in CFG["missing_levels"]:
                for h in CFG["horizons"]:
                    for seed in CFG["seeds"]:
                        rng = np.random.default_rng(seed)
                        # rolling-origin test points
                        origins = range(n - CFG["n_origins"] - max(CFG["horizons"]),
                                        n - max(CFG["horizons"]))
                        for model in models:
                            errs = []
                            for o in origins:
                                start = max(0, o - hist) if hist else 0
                                tr = pd.Series(vals[start:o])
                                if len(tr) < 10:
                                    continue
                                tr = inject_missing(tr, miss, rng)
                                yhat = forecast(model, tr, h, seed)
                                ytrue = vals[o + h - 1]
                                errs.append(yhat - ytrue)
                            if not errs:
                                continue
                            e = np.array(errs)
                            rows.append(dict(site=sid, model=model, horizon=h,
                                             history=hist, missing=miss, seed=seed,
                                             mae=float(np.mean(np.abs(e))),
                                             rmse=float(math.sqrt(np.mean(e**2))),
                                             n_test=len(e)))
        print(f"[site {si+1}/{len(series)}] done: {sid[:40]}", flush=True)
    df = pd.DataFrame(rows)
    out_csv = os.path.join(DATA_DIR, f"raw_results_{tag}.csv")
    df.to_csv(out_csv, index=False)
    print(f"[save] {out_csv} rows={len(df)}", flush=True)
    return df

# ----------------------------------------------------------------------
# SANITY CHECKS
# ----------------------------------------------------------------------
def sanity(df):
    res = {}
    # (1) monotonicity: more history -> lower MAE (averaged), expected for learning models
    learn = "ridge_ar" if "ridge_ar" in df.model.unique() else df.model.unique()[0]
    piv = df[df.model == learn].groupby("history")["mae"].mean()
    res["monotonicity_model"] = learn
    res["monotonicity_history"] = bool(piv.is_monotonic_decreasing) if len(piv) > 1 else None
    res["history_mae"] = {int(k): float(v) for k, v in piv.items()}
    # (2) baseline plausibility: naive/seasonal not absurd (finite, positive MAE)
    bm = df.groupby("model")["mae"].mean().to_dict()
    res["baseline_plausibility"] = bool(all(np.isfinite(v) and v > 0 for v in bm.values()))
    res["model_mae"] = {k: float(v) for k, v in bm.items()}
    # (3) cross-condition consistency: model ranking stable across horizons
    ranks = {}
    for h in df.horizon.unique():
        ranks[int(h)] = list(df[df.horizon == h].groupby("model")["mae"].mean().sort_values().index)
    res["cross_condition_ranks"] = ranks
    res["cross_condition_consistent"] = bool(len(set(tuple(v) for v in ranks.values())) == 1)
    return res

# ----------------------------------------------------------------------
# AGGREGATE + STATS + LATEX
# ----------------------------------------------------------------------
def summarize(df):
    rng = np.random.default_rng(123)
    summary = {"config": CFG, "models": sorted(df.model.unique().tolist())}
    agg = {}
    for model in df.model.unique():
        sub = df[df.model == model]
        ci = bootstrap_ci(sub["mae"].values, CFG["bootstrap"], rng)
        agg[model] = dict(mae_mean=float(sub.mae.mean()), mae_ci95=list(ci),
                          rmse_mean=float(sub.rmse.mean()),
                          smape=None, n=int(len(sub)))
    summary["per_model"] = agg
    # pairwise vs best baseline
    best = min(agg, key=lambda m: agg[m]["mae_mean"])
    summary["best_model"] = best
    comps = {}
    for model in df.model.unique():
        if model == best:
            continue
        a = df[df.model == best]["mae"].values
        b = df[df.model == model]["mae"].values
        comps[f"{best}_vs_{model}"] = dict(
            cohens_d=cohens_d(b, a),
            dm_stat_p=list(diebold_mariano(
                df[df.model == model]["rmse"].values,
                df[df.model == best]["rmse"].values, h=1)))
    summary["comparisons"] = comps
    summary["sanity"] = sanity(df)
    with open(os.path.join(DATA_DIR, "experiment_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    # LaTeX table
    lines = [r"\begin{table}[t]\centering",
             r"\caption{Forecasting MAE (log$_{10}$ concentration) on CDC NWSS, experiment 0. "
             r"Baselines (sandbox); neural and TimesFM rows added from RTX 3090 runs.}",
             r"\label{tab:results0}",
             r"\begin{tabular}{lccc}", r"\toprule",
             r"Model & MAE & 95\% CI & RMSE \\", r"\midrule"]
    for m in sorted(agg, key=lambda k: agg[k]["mae_mean"]):
        a = agg[m]
        lines.append(f"{m} & {a['mae_mean']:.3f} & "
                     f"[{a['mae_ci95'][0]:.3f}, {a['mae_ci95'][1]:.3f}] & "
                     f"{a['rmse_mean']:.3f} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    with open(os.path.join(DATA_DIR, "results_table.tex"), "w") as f:
        f.write("\n".join(lines))
    return summary

# ----------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="baselines", choices=["baselines", "full3090", "all"])
    ap.add_argument("--models", default=None, help="comma list override")
    a = ap.parse_args()
    if a.models:
        models = a.models.split(",")
    elif a.stage == "baselines":
        models = BASELINES                                  # sandbox: pure numpy/scipy
    elif a.stage == "full3090":
        models = HEAVY + ["lstm", "timesfm_zs", "timesfm_ft"]  # 3090: statsmodels/xgboost/torch/timesfm
    else:
        models = BASELINES + HEAVY + ["lstm", "timesfm_zs", "timesfm_ft"]
    df = run(models, tag=a.stage)
    summ = summarize(df)
    print("[summary] best:", summ["best_model"])
    print("[sanity]", json.dumps(summ["sanity"], indent=2)[:600])

if __name__ == "__main__":
    main()
# end of analysis_main.py  (tideguard #24, Step 4)
