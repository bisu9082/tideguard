#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
tideguard (#24) EM&S revision -- INDEPENDENT-PATHOGEN extension (answers R2.2).

Runs the EXACT same rolling-origin benchmark used for SARS-CoV-2 on two further
independent respiratory pathogens already present in the Swiss co-located dataset
(Baumgartner et al. 2025): Influenza B virus (10 cities >=52 wk) and RSV
(8 cities >=52 wk). Influenza A (4 cities) is included as an exploratory panel.

Reuses the same baseline definitions (wbe_forecast.py) and foundation-model
backends (chronos_run.py) as the main pipeline -- nothing new is introduced, so
any difference is attributable to the pathogen, not the method.

Place this file in the tideguard project ROOT (next to wbe_forecast.py,
chronos_run.py) OR run with tideguard root as CWD. Requires the GPU env used for
timesfm_zs / chronos (torch + transformers + chronos-forecasting).

Usage (Ubuntu, GPU env 'tideguard'):
    python pathogen_extension.py --smoke      # 1 config, quick validity check
    python pathogen_extension.py              # full: all pathogens x histories x missing x seeds
Outputs (data/):
    raw_results_pathogens.csv        per (pathogen, city, model, horizon, history, missing, seed)
    alert_perorigin_pathogens.csv    per (pathogen, city, model) surge/pred_rise for decision metrics
    pathogens_summary.json           per-pathogen mean MAE + site-clustered TimesFM-vs-baseline stats
"""
import os, json, argparse, importlib.util
import numpy as np, pandas as pd
import warnings; warnings.filterwarnings("ignore")

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = HERE
# allow running from emss/gpu/ : hop up to project root if needed
if not os.path.exists(os.path.join(ROOT, "wbe_forecast.py")):
    ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
def _imp(name, fn):
    spec = importlib.util.spec_from_file_location(name, os.path.join(ROOT, fn))
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m
wf = _imp("wf", "wbe_forecast.py")
cr = _imp("cr", "chronos_run.py")
CFG = wf.CFG
SWISS = cr.SWISS
DATA = wf.DATA_DIR

BASE = {"naive": wf.f_naive, "seasonal_naive": wf.f_seasonal_naive,
        "ets": wf.f_ets, "ar_ols": wf.f_ar_ols, "ridge_ar": wf.f_ridge_ar}
# pathogen column in the Swiss CSV -> short tag
PATHOGENS = {"Influenza B virus": "fluB", "RSV": "rsv", "Influenza A virus": "fluA"}
TAU, H_SURGE, HIST_SURGE = 0.30, 2, 104

def pathogen_series(col, minw=52):
    d = pd.read_csv(SWISS); d["date"] = pd.to_datetime(d["date"]); out = {}
    for loc, g in d.groupby("location"):
        s = g.set_index("date")[col].resample("W").mean().interpolate(limit=3).dropna()
        s = s[s > 0]
        if len(s) >= minw:
            out[loc] = np.log10(s.values.astype(float))
    return out

def paired_t(percity, a, b):
    d = (percity[a] - percity[b]).dropna()
    if len(d) < 2: return float("nan"), float("nan"), len(d)
    t = d.mean() / (d.std(ddof=1) / np.sqrt(len(d)))
    return float(d.mean()), float(t), int(len(d))

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--fast", action="store_true", help="all cities, single full-history chronological config (minutes)")
    a = ap.parse_args()
    print("device:", cr.DEV, flush=True)
    hmax = max(CFG["horizons"])
    if a.smoke:
        hists, miss, seeds = [52], [0.0], [0]
    elif a.fast:
        hists, miss, seeds = [104], [0.0], [0]      # single full-history chronological config
    else:
        hists, miss, seeds = CFG["histories"], CFG["missing_levels"], CFG["seeds"]
    mae_rows, alert_rows, summary = [], [], {}

    for col, tag in PATHOGENS.items():
        ser = pathogen_series(col, 52)
        if a.smoke: ser = dict(list(ser.items())[:3])
        print(f"\n=== {col} ({tag}): {len(ser)} cities >=52wk -> {list(ser)} ===", flush=True)
        if len(ser) == 0: continue
        for ci, (city, vals) in enumerate(ser.items()):
            vals = np.asarray(vals, float); n = len(vals)
            print(f"  [{tag} {ci+1}/{len(ser)}] {city}: running {len(hists)*len(miss)*len(seeds)} config(s)...", flush=True)
            # ---- MAE benchmark (matched to swiss10_all.py) ----
            origins = [o for o in range(n - CFG["n_origins"] - hmax, n - hmax) if o > 0]
            for hist in hists:
                for ms in miss:
                    for seed in seeds:
                        rng = np.random.default_rng(seed); wins, oks = [], []
                        for o in origins:
                            b = vals[max(0, o - hist):o]
                            if len(b) < 10: continue
                            wins.append(wf.fast_inject(b, ms, rng)); oks.append(o)
                        if not wins: continue
                        tfm = cr.timesfm_predict(wins, hmax)[0]
                        chn = cr.chronos_predict(wins, hmax)[0]
                        for h in CFG["horizons"]:
                            yt = np.array([vals[o + h - 1] for o in oks])
                            rec = {name: float(np.mean(np.abs(
                                     np.array([fn(wins[i], h) for i in range(len(wins))]) - yt)))
                                   for name, fn in BASE.items()}
                            rec["timesfm"] = float(np.mean(np.abs(tfm[:, h - 1] - yt)))
                            rec["chronos"] = float(np.mean(np.abs(chn[:, h - 1] - yt)))
                            for m, v in rec.items():
                                mae_rows.append(dict(pathogen=tag, city=city, model=m, horizon=h,
                                                     history=hist, missing=ms, seed=seed, mae=v, n=len(yt)))
            # ---- per-origin surge alerts (decision metrics; full history, h=2) ----
            so = [o for o in range(n - CFG["n_origins"] - 4, n - 4) if o - 1 >= 10 and o + H_SURGE - 1 < n]
            if so and not a.smoke:
                sw = [vals[max(0, o - HIST_SURGE):o] for o in so]
                cur = np.array([vals[o - 1] for o in so]); tgt = np.array([vals[o + H_SURGE - 1] for o in so])
                surge = (tgt - cur) >= TAU
                tf = cr.timesfm_predict(sw, hmax)[0][:, H_SURGE - 1]
                ch = cr.chronos_predict(sw, hmax)[0][:, H_SURGE - 1]
                for name, fn in BASE.items():
                    pr = np.array([fn(sw[i], H_SURGE) - cur[i] for i in range(len(sw))])
                    for i in range(len(sw)):
                        alert_rows.append((tag, city, name, float(pr[i]), bool(surge[i])))
                for i in range(len(sw)):
                    alert_rows.append((tag, city, "timesfm", float(tf[i] - cur[i]), bool(surge[i])))
                    alert_rows.append((tag, city, "chronos", float(ch[i] - cur[i]), bool(surge[i])))
            print(f"  [{tag} {ci+1}/{len(ser)}] {city} done", flush=True)

    df = pd.DataFrame(mae_rows); df.to_csv(os.path.join(DATA, "raw_results_pathogens.csv"), index=False)
    print("\n[saved] raw_results_pathogens.csv rows", len(df), flush=True)
    if alert_rows:
        pd.DataFrame(alert_rows, columns=["pathogen","city","model","pred_rise","surge"]).to_csv(
            os.path.join(DATA, "alert_perorigin_pathogens.csv"), index=False)
        print("[saved] alert_perorigin_pathogens.csv rows", len(alert_rows), flush=True)
    # per-pathogen summary
    for tag in set(df.pathogen):
        sub = df[df.pathogen == tag]
        percity = sub.groupby(["city", "model"]).mae.mean().unstack()
        stats = {}
        for base in ["naive", "ridge_ar", "ets", "seasonal_naive"]:
            if base in percity: 
                md, t, nn = paired_t(percity, "timesfm", base); stats[f"timesfm_vs_{base}"] = dict(dMAE=round(md,4), t=round(t,3), n=nn)
        summary[tag] = dict(cities=int(percity.shape[0]),
                            mean_mae=percity.mean().round(4).sort_values().to_dict(), stats=stats)
        print(f"\n=== {tag}: mean MAE over cities ===\n", percity.mean().round(4).sort_values().to_string(), flush=True)
    json.dump(summary, open(os.path.join(DATA, "pathogens_summary.json"), "w"), indent=2)
    print("\n[done] pathogens_summary.json", flush=True)

if __name__ == "__main__":
    main()
