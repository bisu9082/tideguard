#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
tideguard (#24) revision #3 — expand the Swiss cross-region panel from n=4 to
n=10 municipal SARS-CoV-2 series (>=52 weekly points) and recompute RQ3 with
MATCHED baselines and the two foundation models on identical rolling origins,
so the site-clustered comparison uses 10 resampling units instead of 4.

Reuses the exact baseline definitions (wbe_forecast.py) and FM backends
(chronos_run.py) already used elsewhere, so nothing new is introduced.

Run (GPU env 'tideguard'):
    python swiss10_all.py --smoke     # 3 cities, 1 config — quick validation
    python swiss10_all.py             # full: 10 cities x histories x missing x seeds
Out:
    data/raw_results_swiss10.csv      # per (city, model, horizon, history, missing, seed)
    data/swiss10_summary.json         # per-city mean MAE + site-clustered stats
"""
import os, json, argparse, importlib.util
import numpy as np, pandas as pd
import warnings; warnings.filterwarnings("ignore")

HERE = os.path.dirname(os.path.abspath(__file__))
def _imp(name, fn):
    spec = importlib.util.spec_from_file_location(name, os.path.join(HERE, fn))
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m
wf = _imp("wf", "wbe_forecast.py")
cr = _imp("cr", "chronos_run.py")
CFG = wf.CFG
SWISS = cr.SWISS

# exact baseline definitions reused from the main pipeline
BASE = {"naive": wf.f_naive, "seasonal_naive": wf.f_seasonal_naive,
        "ets": wf.f_ets, "ar_ols": wf.f_ar_ols, "ridge_ar": wf.f_ridge_ar}

def swiss_series(minw=52):
    d = pd.read_csv(SWISS); d["date"] = pd.to_datetime(d["date"]); out = {}
    for loc, g in d.groupby("location"):
        s = g.set_index("date")["SARS-CoV-2"].resample("W").mean().interpolate(limit=3).dropna()
        s = s[s > 0]
        if len(s) >= minw:
            out[loc] = np.log10(s.values.astype(float))
    return out

def paired_t(percity, a, b):
    d = (percity[a] - percity[b]).dropna()
    t = d.mean() / (d.std(ddof=1) / np.sqrt(len(d)))
    return float(d.mean()), float(t), int(len(d))

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    print("device:", cr.DEV, flush=True)
    ser = swiss_series(52)
    if a.smoke:
        ser = dict(list(ser.items())[:3])
    print("Swiss cities (>=52 wk):", len(ser), list(ser), flush=True)

    hmax = max(CFG["horizons"])
    hists = [52] if a.smoke else CFG["histories"]
    miss  = [0.0] if a.smoke else CFG["missing_levels"]
    seeds = [0]   if a.smoke else CFG["seeds"]

    rows = []
    for ci, (city, vals) in enumerate(ser.items()):
        vals = np.asarray(vals, float); n = len(vals)
        origins = [o for o in range(n - CFG["n_origins"] - hmax, n - hmax) if o > 0]
        for hist in hists:
            for ms in miss:
                for seed in seeds:
                    rng = np.random.default_rng(seed)
                    wins, oks = [], []
                    for o in origins:
                        b = vals[max(0, o - hist):o]
                        if len(b) < 10:
                            continue
                        wins.append(wf.fast_inject(b, ms, rng)); oks.append(o)
                    if not wins:
                        continue
                    tfm = cr.timesfm_predict(wins, hmax)[0]    # [N, hmax]
                    chn = cr.chronos_predict(wins, hmax)[0]
                    for h in CFG["horizons"]:
                        yt = np.array([vals[o + h - 1] for o in oks])
                        rec = {name: float(np.mean(np.abs(
                                   np.array([fn(wins[i], h) for i in range(len(wins))]) - yt)))
                               for name, fn in BASE.items()}
                        rec["timesfm"] = float(np.mean(np.abs(tfm[:, h - 1] - yt)))
                        rec["chronos"] = float(np.mean(np.abs(chn[:, h - 1] - yt)))
                        for m, v in rec.items():
                            rows.append(dict(city=city, model=m, horizon=h, history=hist,
                                             missing=ms, seed=seed, mae=v, n=len(yt)))
        print(f"[{ci+1}/{len(ser)}] {city} done", flush=True)

    df = pd.DataFrame(rows)
    out = os.path.join(cr.DATA, "raw_results_swiss10.csv"); df.to_csv(out, index=False)
    print("[saved]", out, "rows", len(df), flush=True)

    percity = df.groupby(["city", "model"]).mae.mean().unstack()
    print("\n=== Per-city mean MAE (log10) ===\n", percity.round(3).to_string(), flush=True)
    print("\n=== Mean over cities ===\n", percity.mean().round(4).sort_values().to_string(), flush=True)
    stats = {}
    for base in ["naive", "ridge_ar", "ets"]:
        md, t, nn = paired_t(percity, "timesfm", base)
        stats[f"timesfm_vs_{base}"] = dict(dMAE=round(md, 4), t=round(t, 3), n=nn)
        print(f"site-clustered TimesFM vs {base}: dMAE={md:+.4f}  t={t:+.2f}  n={nn}", flush=True)
    json.dump({"cities": list(ser),
               "mean_mae": percity.mean().round(4).to_dict(),
               "stats": stats},
              open(os.path.join(cr.DATA, "swiss10_summary.json"), "w"), indent=2)
    print("[done]", flush=True)

if __name__ == "__main__":
    main()
