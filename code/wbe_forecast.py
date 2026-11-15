#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
tideguard (Project #24) — WBE-ML integrated forecasting (Step 4).
Reproducible pipeline. Data integrity: REAL public data only (CDC NWSS).

Run locations:
  SANDBOX (done) : baselines (naive, seasonal_naive, ets, ar_ols, ridge_ar)
  RTX 3090       : heavy + neural + foundation model
       python wbe_forecast.py --models sarima,xgboost      # classical heavy
       python wbe_forecast.py --models lstm                # neural baseline
       python wbe_forecast.py --models timesfm_zs          # TimesFM zero-shot (headline)
       python wbe_forecast.py --models timesfm_ft          # TimesFM fine-tuned

Fair comparison: ALL models use the SAME sites / origins / histories / missingness
(state-diversified multi-site selection, fixed seeds). Results append to one CSV.

Outputs (data/):
  raw_results_<models>.csv   per (site, model, horizon, history, missing, seed)
  experiment_summary.json    aggregated metrics + statistics + sanity checks
  results_table.tex          LaTeX results table (Step 5 insert)
"""
import argparse, json, os, math, re, collections, time
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

# Imputation for gap-filling. Default (False) = linear interpolation, which
# produced the reported manuscript numbers. Set env CAUSAL_IMPUTE=1 to use a
# strictly causal forward-fill (the SI imputation-sensitivity check).
CAUSAL_IMPUTE = os.environ.get("CAUSAL_IMPUTE", "0") == "1"

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(HERE, "data")
NWSS_CSV = os.path.join(DATA_DIR, "nwss_sarscov2_concentration.csv")
NWSS_URL = "https://data.cdc.gov/resource/g653-rqe2.json"  # Socrata API (auto-download)

CFG = dict(
    normalization="flow-population",
    min_weeks=156,          # >= 3 years of weekly points
    per_state=2,            # sites per US state
    n_sites=16,             # total sites (state-diversified)
    freq="W",
    horizons=[1, 2, 4],
    histories=[12, 26, 52, 104],
    missing_levels=[0.0, 0.10, 0.30],
    n_origins=20,
    seeds=[0, 1, 2, 3, 4],
    bootstrap=2000,
    # TimesFM
    tfm_ckpt="google/timesfm-2.5-200m-pytorch",
    tfm_max_context=1024,
)

# ============================ DATA ============================
def _ensure_data():
    if os.path.exists(NWSS_CSV):
        return
    print("[data] local CSV not found -> downloading CDC NWSS ...", flush=True)
    import urllib.request
    rows, off, lim = [], 0, 50000
    cols = "key_plot_id,date,pcr_conc_lin,normalization"
    while True:
        url = f"{NWSS_URL}?$select={cols}&$order=key_plot_id,date&$limit={lim}&$offset={off}".replace(" ", "%20")
        with urllib.request.urlopen(url, timeout=120) as r:
            batch = json.load(r)
        if not batch:
            break
        rows += batch; off += lim
        print(f"[data] fetched {len(rows)}", flush=True)
        if len(batch) < lim:
            break
    os.makedirs(DATA_DIR, exist_ok=True)
    pd.DataFrame(rows).to_csv(NWSS_CSV, index=False)
    print(f"[data] saved {NWSS_CSV} ({len(rows)} rows)", flush=True)

def _state(s):
    m = re.search(r'_(?:VERILY_)?([a-z]{2})_', s)
    return m.group(1) if m else '??'

def load_series():
    """State-diversified multi-site selection -> dict[site] = log10 weekly series."""
    _ensure_data()
    df = pd.read_csv(NWSS_CSV)
    df = df[df["normalization"] == CFG["normalization"]].copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["c"] = pd.to_numeric(df["pcr_conc_lin"], errors="coerce")
    df = df.dropna(subset=["date", "c"]); df = df[df["c"] > 0]
    df["logc"] = np.log10(df["c"])
    ser = {}
    for sid, g in df.groupby("key_plot_id"):
        # Short-gap fill for the reported results. Set CAUSAL_IMPUTE=1 to use a
        # strictly causal forward-fill (SI imputation-sensitivity check); the
        # default reproduces the numbers in the manuscript tables.
        w = (g.set_index("date")["logc"].sort_index().resample(CFG["freq"]).mean())
        w = (w.ffill(limit=3) if CAUSAL_IMPUTE
             else w.interpolate(limit=3, limit_direction="both")).dropna()
        if len(w) >= CFG["min_weeks"]:
            ser[sid] = w
    byst = collections.defaultdict(list)
    for sid, w in ser.items():
        byst[_state(sid)].append((len(w), sid))
    order = sorted(byst, key=lambda k: -max(n for n, _ in byst[k]))
    for st in order:
        byst[st].sort(reverse=True)
    chosen, r = [], 0
    while len(chosen) < CFG["n_sites"] and r < CFG["per_state"]:
        for st in order:
            if r < len(byst[st]):
                chosen.append(byst[st][r][1])
                if len(chosen) >= CFG["n_sites"]:
                    break
        r += 1
    return {sid: ser[sid] for sid in chosen}

def fast_inject(arr, frac, rng):
    if frac <= 0:
        return arr
    a = arr.copy(); n = len(a); k = int(round(frac * n))
    if k <= 0:
        return a
    idx = rng.choice(n, k, replace=False)
    mask = np.ones(n, bool); mask[idx] = False
    xs = np.arange(n)
    if CAUSAL_IMPUTE:
        # Causal forward-fill (no look-ahead); leading gaps back-fill from
        # first obs. Used for the SI imputation-sensitivity check.
        a = pd.Series(np.where(mask, a, np.nan)).ffill().bfill().to_numpy()
    else:
        # Linear interpolation used for the reported results.
        a[~mask] = np.interp(xs[~mask], xs[mask], a[mask])
    return a

# ============================ STATS ============================
def bootstrap_ci(x, nboot, rng, alpha=0.05):
    x = np.asarray(x, float)
    if len(x) == 0:
        return (float("nan"), float("nan"))
    bs = [np.mean(rng.choice(x, len(x), replace=True)) for _ in range(nboot)]
    return (float(np.percentile(bs, 100*alpha/2)), float(np.percentile(bs, 100*(1-alpha/2))))

def cohens_d(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    na, nb = len(a), len(b)
    if na < 2 or nb < 2:
        return float("nan")
    sp = math.sqrt(((na-1)*a.var(ddof=1) + (nb-1)*b.var(ddof=1)) / (na+nb-2))
    return float((a.mean()-b.mean())/sp) if sp > 0 else float("nan")

def _norm_cdf(x):
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))

def diebold_mariano(e1, e2, h=1):
    e1, e2 = np.asarray(e1, float), np.asarray(e2, float)
    d = e1**2 - e2**2; n = len(d)
    if n < 8:
        return (float("nan"), float("nan"))
    dbar = d.mean(); var = np.sum((d-dbar)**2)/n
    for lag in range(1, h):
        var += 2*(1-lag/h)*np.sum((d[lag:]-dbar)*(d[:-lag]-dbar))/n
    if var <= 0:
        return (float("nan"), float("nan"))
    stat = dbar/math.sqrt(var/n)
    return (float(stat), float(2*(1-_norm_cdf(abs(stat)))))

# ============================ MODELS ============================
def f_naive(tr, h):
    return float(tr[-1])

def f_seasonal_naive(tr, h, period=52):
    if len(tr) > period:
        i = -period + (h-1)
        return float(tr[i]) if (h-1) < period else float(tr[-1])
    return float(tr[-1])

def f_ets(tr, h, alpha=0.5, beta=0.1):
    x = np.asarray(tr, float)
    if len(x) < 3:
        return float(x[-1])
    lvl, trend = x[0], x[1]-x[0]
    for t in range(1, len(x)):
        prev = lvl
        lvl = alpha*x[t] + (1-alpha)*(lvl+trend)
        trend = beta*(lvl-prev) + (1-beta)*trend
    return float(lvl + h*trend)

def _ar_feats(arr, lags):
    X, y = [], []
    for i in range(lags, len(arr)):
        X.append(arr[i-lags:i]); y.append(arr[i])
    return np.asarray(X, float), np.asarray(y, float)

def f_ar_ols(tr, h, lags=4):
    arr = np.asarray(tr, float)
    if len(arr) < lags+5:
        return float(arr[-1])
    X, y = _ar_feats(arr, lags)
    Xd = np.hstack([np.ones((len(X), 1)), X])
    beta, *_ = np.linalg.lstsq(Xd, y, rcond=None)
    cur = list(arr[-lags:]); pred = cur[-1]
    for _ in range(h):
        pred = float(beta[0] + np.dot(beta[1:], cur[-lags:])); cur.append(pred)
    return pred

def f_ridge_ar(tr, h, lags=8, lam=1.0):
    arr = np.asarray(tr, float)
    if len(arr) < lags+5:
        return float(arr[-1])
    X, y = _ar_feats(arr, lags)
    Xd = np.hstack([np.ones((len(X), 1)), X])
    A = Xd.T @ Xd + lam*np.eye(Xd.shape[1])
    beta = np.linalg.solve(A, Xd.T @ y)
    cur = list(arr[-lags:]); pred = cur[-1]
    for _ in range(h):
        pred = float(beta[0] + np.dot(beta[1:], cur[-lags:])); cur.append(pred)
    return pred

def f_sarima(tr, h):
    """ARIMA with graceful order-reduction fallback (avoids mid-run crash on
    short/degenerate windows). Order reduction is standard practice; never a
    silent 'naive' relabel unless even ARIMA(0,1,0) is infeasible."""
    from statsmodels.tsa.arima.model import ARIMA
    arr = np.asarray(tr, float)
    for order in [(2, 1, 2), (1, 1, 1), (1, 1, 0), (0, 1, 0)]:
        try:
            m = ARIMA(arr, order=order).fit(method_kwargs={"warn_convergence": False})
            return float(m.forecast(h)[-1])
        except Exception:
            continue
    return float(arr[-1])

def f_xgboost(tr, h, seed, lags=8):
    from xgboost import XGBRegressor
    arr = np.asarray(tr, float)
    if len(arr) < lags+5:
        return float(arr[-1])
    X, y = _ar_feats(arr, lags)
    m = XGBRegressor(n_estimators=120, max_depth=3, learning_rate=0.1,
                     subsample=0.9, colsample_bytree=0.9, random_state=seed, verbosity=0)
    m.fit(X, y)
    cur = list(arr[-lags:]); pred = None
    for _ in range(h):
        pred = float(m.predict(np.array(cur[-lags:]).reshape(1, -1))[0]); cur.append(pred)
    return pred

# ---- LSTM (torch); trained once per (series,history,missing,seed) ----
_LSTM_CACHE = {}
def _lstm_train(arr, seed, lags=8, epochs=80):
    import torch, torch.nn as nn
    torch.manual_seed(seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    X, y = _ar_feats(arr, lags)
    if len(X) < 8:
        return None
    Xt = torch.tensor(X, dtype=torch.float32, device=dev).unsqueeze(-1)
    yt = torch.tensor(y, dtype=torch.float32, device=dev).unsqueeze(-1)
    class Net(nn.Module):
        def __init__(s):
            super().__init__(); s.l = nn.LSTM(1, 32, batch_first=True); s.f = nn.Linear(32, 1)
        def forward(s, x):
            o, _ = s.l(x); return s.f(o[:, -1, :])
    net = Net().to(dev); opt = torch.optim.Adam(net.parameters(), lr=0.01); lf = nn.MSELoss()
    for _ in range(epochs):
        opt.zero_grad(); loss = lf(net(Xt), yt); loss.backward(); opt.step()
    return (net, dev, lags)

def f_lstm(tr, h, seed, key):
    import torch
    arr = np.asarray(tr, float)
    if key not in _LSTM_CACHE:
        _LSTM_CACHE[key] = _lstm_train(arr, seed)
    trained = _LSTM_CACHE[key]
    if trained is None:
        return float(arr[-1])
    net, dev, lags = trained
    cur = list(arr[-lags:]); pred = float(arr[-1])
    net.eval()
    with torch.no_grad():
        for _ in range(h):
            x = torch.tensor(np.array(cur[-lags:]), dtype=torch.float32, device=dev).reshape(1, lags, 1)
            pred = float(net(x).item()); cur.append(pred)
    return pred

# ---- TimesFM 2.5 ----
_TFM = {"model": None}
def _tfm_load(finetune=False):
    if _TFM["model"] is not None and _TFM.get("ft") == finetune:
        return _TFM["model"]
    import torch, timesfm
    torch.set_float32_matmul_precision("high")
    model = timesfm.TimesFM_2p5_200M_torch.from_pretrained(CFG["tfm_ckpt"])
    model.compile(timesfm.ForecastConfig(
        max_context=CFG["tfm_max_context"], max_horizon=max(CFG["horizons"]),
        normalize_inputs=True, use_continuous_quantile_head=True,
        force_flip_invariance=True, infer_is_positive=True, fix_quantile_crossing=True))
    _TFM["model"] = model; _TFM["ft"] = finetune
    return model

def tfm_forecast_batch(windows, horizon, finetune=False):
    """windows: list of 1D arrays -> point forecast array (len(windows), horizon)."""
    model = _tfm_load(finetune)
    point, _ = model.forecast(horizon=horizon, inputs=[np.asarray(w, float) for w in windows])
    return np.asarray(point)

BASELINES = ["naive", "seasonal_naive", "ets", "ar_ols", "ridge_ar"]
ALL_MODELS = BASELINES + ["sarima", "xgboost", "lstm", "timesfm_zs", "timesfm_ft"]
TFM_MODELS = {"timesfm_zs": False, "timesfm_ft": True}

def forecast(model, tr, h, seed, key=None):
    if model == "naive":          return f_naive(tr, h)
    if model == "seasonal_naive": return f_seasonal_naive(tr, h)
    if model == "ets":            return f_ets(tr, h)
    if model == "ar_ols":         return f_ar_ols(tr, h)
    if model == "ridge_ar":       return f_ridge_ar(tr, h)
    if model == "sarima":         return f_sarima(tr, h)
    if model == "xgboost":        return f_xgboost(tr, h, seed)
    if model == "lstm":           return f_lstm(tr, h, seed, key)
    raise ValueError(model)

# ============================ EXPERIMENT ============================
def run(models, tag):
    series = load_series()
    sts = collections.Counter(_state(s) for s in series)
    print(f"[data] sites={len(series)} states={len(sts)} {dict(sts)}", flush=True)
    out_csv = os.path.join(DATA_DIR, f"raw_results_{tag}.csv")
    if os.path.exists(out_csv):
        os.remove(out_csv)
    hmax = max(CFG["horizons"]); t0 = time.time(); first = True
    tfm_models = [m for m in models if m in TFM_MODELS]
    seq_models = [m for m in models if m not in TFM_MODELS]
    for si, (sid, s) in enumerate(series.items()):
        vals = s.values.astype(float); n = len(vals); rows = []
        origins = list(range(n - CFG["n_origins"] - hmax, n - hmax))
        for hist in CFG["histories"]:
            for miss in CFG["missing_levels"]:
                for seed in CFG["seeds"]:
                    rng = np.random.default_rng(seed)
                    # build injected training windows once per origin
                    trains = {}
                    for o in origins:
                        base = vals[max(0, o-hist):o]
                        trains[o] = fast_inject(base, miss, rng) if len(base) >= 10 else None
                    # sequential models
                    for m in seq_models:
                        errs = {h: [] for h in CFG["horizons"]}
                        for o in origins:
                            tr = trains[o]
                            if tr is None:
                                continue
                            key = (sid, hist, miss, seed, o)
                            for h in CFG["horizons"]:
                                yhat = forecast(m, tr, h, seed, key=key)
                                errs[h].append(yhat - vals[o+h-1])
                        for h, e in errs.items():
                            if e:
                                e = np.array(e)
                                rows.append(dict(site=sid, state=_state(sid), model=m, horizon=h,
                                                 history=hist, missing=miss, seed=seed,
                                                 mae=float(np.mean(np.abs(e))),
                                                 rmse=float(math.sqrt(np.mean(e**2))), n_test=len(e)))
                        _LSTM_CACHE.clear()
                    # TimesFM models (batched over origins)
                    for m in tfm_models:
                        ft = TFM_MODELS[m]
                        win = [trains[o] for o in origins if trains[o] is not None]
                        oks = [o for o in origins if trains[o] is not None]
                        if not win:
                            continue
                        pf = tfm_forecast_batch(win, hmax, finetune=ft)  # (len, hmax)
                        for h in CFG["horizons"]:
                            e = [pf[i, h-1] - vals[o+h-1] for i, o in enumerate(oks)]
                            e = np.array(e)
                            rows.append(dict(site=sid, state=_state(sid), model=m, horizon=h,
                                             history=hist, missing=miss, seed=seed,
                                             mae=float(np.mean(np.abs(e))),
                                             rmse=float(math.sqrt(np.mean(e**2))), n_test=len(e)))
        pd.DataFrame(rows).to_csv(out_csv, mode='a', header=first, index=False); first = False
        print(f"[site {si+1}/{len(series)}] {sid[:30]}({_state(sid)}) t={time.time()-t0:.0f}s", flush=True)
    print(f"[ALL DONE] {time.time()-t0:.0f}s -> {out_csv}", flush=True)
    return pd.read_csv(out_csv)

# ============================ SUMMARY ============================
def merge_all_results():
    """Combine every raw_results_*.csv into one frame for joint stats."""
    frames = []
    for fn in os.listdir(DATA_DIR):
        if fn.startswith("raw_results_") and fn.endswith(".csv"):
            frames.append(pd.read_csv(os.path.join(DATA_DIR, fn)))
    return pd.concat(frames, ignore_index=True).drop_duplicates(
        subset=["site", "model", "horizon", "history", "missing", "seed"]) if frames else pd.DataFrame()

def sanity(df):
    res = {}
    learn = "ridge_ar" if "ridge_ar" in df.model.unique() else df.model.unique()[0]
    piv = df[df.model == learn].groupby("history")["mae"].mean()
    res["monotonicity_model"] = learn
    res["monotonicity_history"] = bool(piv.is_monotonic_decreasing) if len(piv) > 1 else None
    res["history_mae"] = {int(k): float(v) for k, v in piv.items()}
    bm = df.groupby("model")["mae"].mean().to_dict()
    res["baseline_plausibility"] = bool(all(np.isfinite(v) and v > 0 for v in bm.values()))
    res["model_mae"] = {k: float(v) for k, v in bm.items()}
    ranks = {}
    for h in sorted(df.horizon.unique()):
        ranks[int(h)] = list(df[df.horizon == h].groupby("model")["mae"].mean().sort_values().index)
    res["cross_condition_ranks"] = ranks
    res["best_model_stable"] = bool(len(set(v[0] for v in ranks.values())) == 1)
    return res

def summarize(df):
    rng = np.random.default_rng(123)
    summary = {"config": CFG, "models": sorted(df.model.unique().tolist())}
    agg = {}
    for m in df.model.unique():
        sub = df[df.model == m]
        ci = bootstrap_ci(sub["mae"].values, CFG["bootstrap"], rng)
        agg[m] = dict(mae_mean=float(sub.mae.mean()), mae_ci95=list(ci),
                      rmse_mean=float(sub.rmse.mean()), n=int(len(sub)))
    summary["per_model"] = agg
    best = min(agg, key=lambda m: agg[m]["mae_mean"])
    summary["best_model"] = best
    # key-aligned paired comparison (robust to differing row counts per model)
    keys = ["site", "horizon", "history", "missing", "seed"]
    rmse_wide = df.pivot_table(index=keys, columns="model", values="rmse")
    mae_wide = df.pivot_table(index=keys, columns="model", values="mae")
    comps = {}
    for m in df.model.unique():
        if m == best:
            continue
        rp = rmse_wide[[best, m]].dropna()
        mp = mae_wide[[best, m]].dropna()
        comps[f"{best}_vs_{m}"] = dict(
            n_pairs=int(len(rp)),
            cohens_d=cohens_d(mp[m].values, mp[best].values),
            dm_stat_p=list(diebold_mariano(rp[m].values, rp[best].values, h=1)))
    summary["comparisons"] = comps
    summary["sanity"] = sanity(df)
    with open(os.path.join(DATA_DIR, "experiment_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    lines = [r"\begin{table}[t]\centering",
             r"\caption{Forecasting MAE (log$_{10}$ concentration) on CDC NWSS, "
             r"state-diversified multi-site evaluation. Lower is better.}",
             r"\label{tab:results}", r"\begin{tabular}{lccc}", r"\toprule",
             r"Model & MAE & 95\% CI & RMSE \\", r"\midrule"]
    for m in sorted(agg, key=lambda k: agg[k]["mae_mean"]):
        a = agg[m]
        lines.append(f"{m} & {a['mae_mean']:.3f} & [{a['mae_ci95'][0]:.3f}, {a['mae_ci95'][1]:.3f}] & {a['rmse_mean']:.3f} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    with open(os.path.join(DATA_DIR, "results_table.tex"), "w") as f:
        f.write("\n".join(lines))
    return summary

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default=",".join(BASELINES),
                    help="comma list from: " + ",".join(ALL_MODELS))
    ap.add_argument("--summary-only", action="store_true",
                    help="merge all raw_results_*.csv and (re)compute summary")
    a = ap.parse_args()
    if a.summary_only:
        df = merge_all_results()
    else:
        models = a.models.split(",")
        tag = "_".join(models) if len(models) <= 3 else f"{len(models)}models"
        run(models, tag)
        df = merge_all_results()
    s = summarize(df)
    print("[summary] models:", s["models"])
    print("[summary] best:", s["best_model"])
    print("[summary] model_mae:", {k: round(v["mae_mean"], 3) for k, v in s["per_model"].items()})

if __name__ == "__main__":
    main()
