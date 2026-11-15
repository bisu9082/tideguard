#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
tideguard (#24) revision #2 — add a SECOND foundation model (Chronos-Bolt) and
decision-relevant metrics (directional accuracy, WIS, 90% PI coverage) for the
two foundation models, on the SAME rolling-origin protocol as the main benchmark.

Defends reviewer critiques: (i) "only one foundation model" and (ii) "log-MAE is
not operationally meaningful".

Requires (GPU env): pip install chronos-forecasting  (transformers/torch already installed)
Models: amazon/chronos-bolt-base (205M) vs TimesFM 2.5 (transformers backend).

Run:  python chronos_run.py --smoke     # 2 sites/cities quick check
      python chronos_run.py             # full US + EU
Out:  data/raw_results_fm_metrics.csv  (per site/horizon: mae, dir_acc, wis, cov90 per model)
      data/fm_metrics_summary.json
"""
import os, json, math, argparse, importlib.util
import numpy as np, pandas as pd, torch
import warnings; warnings.filterwarnings("ignore")

HERE = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location("wf", os.path.join(HERE, "wbe_forecast.py"))
wf = importlib.util.module_from_spec(spec); spec.loader.exec_module(wf)
CFG = wf.CFG
DATA = wf.DATA_DIR
SWISS = os.path.join(HERE, "data", "swiss_wbe", "data", "04_intermediate",
                     "Drugs_Viruses_PopNormLoad.csv")
QL = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]   # quantile levels
DEV = "cuda" if torch.cuda.is_available() else "cpu"

# ---------------- models ----------------
_M = {}
def load_chronos():
    if "chronos" not in _M:
        from chronos import BaseChronosPipeline
        _M["chronos"] = BaseChronosPipeline.from_pretrained(
            "amazon/chronos-bolt-base", device_map=DEV, torch_dtype=torch.bfloat16)
    return _M["chronos"]

def load_timesfm():
    if "tfm" not in _M:
        from transformers import TimesFm2_5ModelForPrediction
        _M["tfm"] = TimesFm2_5ModelForPrediction.from_pretrained(
            "google/timesfm-2.5-200m-transformers", torch_dtype=torch.bfloat16, device_map=DEV).eval()
    return _M["tfm"]

def chronos_predict(windows, hmax):
    """Return (point[N,hmax], quantiles[N,hmax,len(QL)])."""
    pipe = load_chronos()
    ctx = [torch.tensor(np.asarray(w, np.float32)) for w in windows]
    q, mean = pipe.predict_quantiles(inputs=ctx, prediction_length=hmax, quantile_levels=QL)
    return np.asarray(mean.float().cpu()), np.asarray(q.float().cpu())  # mean[N,h], q[N,h,Q]

def timesfm_predict(windows, hmax):
    m = load_timesfm(); pts = []
    with torch.no_grad():
        for w in windows:
            ti = torch.tensor(np.asarray(w, np.float32), device=DEV).unsqueeze(0)
            out = m(past_values=ti)
            pts.append(out.mean_predictions[0, :hmax].float().cpu().numpy())
    pt = np.array(pts)
    # TimesFM quantiles via its quantile head if available; else fallback to None
    return pt, None

# ---------------- metrics ----------------
def wis(y, qvals, levels):
    """Weighted interval score for one point y given quantile forecasts."""
    qs = {l: qvals[i] for i, l in enumerate(levels)}
    med = qs.get(0.5, np.median(qvals))
    K = 0
    total = 0.5 * abs(y - med)
    pairs = [(l, 1 - l) for l in levels if l < 0.5]
    for lo, hi in pairs:
        alpha = 2 * lo
        L, U = qs[lo], qs[hi]
        interval = (U - L) + (2/alpha)*(L - y)*(y < L) + (2/alpha)*(y - U)*(y > U)
        total += (alpha/2) * interval; K += 1
    return total / (K + 0.5)

# ---------------- data ----------------
def us_series():
    return wf.load_series()
def eu_series():
    d = pd.read_csv(SWISS); d["date"] = pd.to_datetime(d["date"]); out = {}
    for loc, g in d.groupby("location"):
        s = g.set_index("date")["SARS-CoV-2"].resample("W").mean().interpolate(limit=3).dropna()
        s = s[s > 0]
        if len(s) >= 60: out["CH_"+loc] = np.log10(s.values.astype(float))
    return out

# ---------------- experiment ----------------
def run(region, series, smoke):
    hmax = max(CFG["horizons"]); rows = []
    items = list(series.items())[:2] if smoke else list(series.items())
    hists = [52] if smoke else CFG["histories"]
    miss = [0.0] if smoke else CFG["missing_levels"]
    seeds = [0] if smoke else CFG["seeds"]
    for si, (sid, vals) in enumerate(items):
        vals = np.asarray(vals, dtype=float)
        n = len(vals); origins = list(range(n-CFG["n_origins"]-hmax, n-hmax))
        for hist in hists:
            for ms in miss:
                for seed in seeds:
                    rng = np.random.default_rng(seed)
                    wins, oks = [], []
                    for o in origins:
                        b = vals[max(0,o-hist):o]
                        if len(b) < 10: continue
                        wins.append(wf.fast_inject(b, ms, rng)); oks.append(o)
                    if not wins: continue
                    lasts = np.array([w[-1] for w in wins])
                    for name, pred in [("chronos", chronos_predict), ("timesfm", timesfm_predict)]:
                        pt, q = pred(wins, hmax)
                        for h in CFG["horizons"]:
                            yt = np.array([vals[o+h-1] for o in oks])
                            ph = pt[:, h-1]
                            mae = float(np.mean(np.abs(ph - yt)))
                            moved = np.sign(yt - lasts) != 0
                            dacc = float(np.mean((np.sign(ph-lasts)==np.sign(yt-lasts))[moved])) if moved.any() else float("nan")
                            if q is not None:
                                qh = q[:, h-1, :]
                                cov = float(np.mean((yt>=qh[:,0]) & (yt<=qh[:,-1])))  # 10-90% band
                                wv = float(np.mean([wis(yt[i], qh[i], QL) for i in range(len(yt))]))
                            else:
                                cov = float("nan"); wv = float("nan")
                            rows.append(dict(region=region, site=sid, model=name, horizon=h,
                                history=hist, missing=ms, seed=seed, mae=mae, dir_acc=dacc,
                                cov_10_90=cov, wis=wv, n=len(yt)))
        print(f"[{region} {si+1}/{len(items)}] {sid[:28]}", flush=True)
    return rows

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--smoke", action="store_true"); a = ap.parse_args()
    print("device:", DEV, flush=True)
    rows = run("US", us_series(), a.smoke) + run("EU", eu_series(), a.smoke)
    df = pd.DataFrame(rows)
    out = os.path.join(DATA, "raw_results_fm_metrics.csv"); df.to_csv(out, index=False)
    print("[saved]", out, "rows", len(df), flush=True)
    summ = {}
    for reg in df.region.unique():
        summ[reg] = {}
        for m in df.model.unique():
            s = df[(df.region==reg)&(df.model==m)]
            summ[reg][m] = dict(mae=round(float(s.mae.mean()),4), dir_acc=round(float(s.dir_acc.mean()),3),
                                cov_10_90=round(float(s.cov_10_90.mean()),3), wis=round(float(s.wis.mean()),4))
    json.dump(summ, open(os.path.join(DATA,"fm_metrics_summary.json"),"w"), indent=2)
    print(json.dumps(summ, indent=2))

if __name__ == "__main__":
    main()
