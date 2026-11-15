#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
tideguard (#24) RQ3 — does TimesFM zero-shot superiority GENERALIZE across regions?
US (CDC NWSS, already done) vs EUROPE (Switzerland, 4 dense cities).

Region 2 (Europe): Swiss SARS-CoV-2 weekly load per city, from the already-downloaded
ERIC package (data/swiss_wbe/.../Drugs_Viruses_PopNormLoad.csv). No new download.

Reuses validated model functions from wbe_forecast.py. Same rolling-origin protocol,
same ablations (history length, missingness), same seeds -> directly comparable to US.

Baselines run anywhere (numpy). timesfm_zs runs only where the `timesfm` pkg + torch
are available (Ku's GPU). Output: data/raw_results_eu_<tag>.csv + data/rq3_summary.json
"""
import os, json, math, importlib.util
import numpy as np, pandas as pd
import warnings; warnings.filterwarnings("ignore")

HERE = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location("wf", os.path.join(HERE, "wbe_forecast.py"))
wf = importlib.util.module_from_spec(spec); spec.loader.exec_module(wf)
CFG = wf.CFG
SWISS = os.path.join(HERE, "data", "swiss_wbe", "data", "04_intermediate",
                     "Drugs_Viruses_PopNormLoad.csv")
TARGET = "SARS-CoV-2"
MIN_WEEKS = 60

_TFB = {"m": None}
def _tf_zs_predict(windows, hmax):
    """TimesFM 2.5 zero-shot via transformers backend (GPU). Returns (N,hmax)."""
    import torch
    from transformers import TimesFm2_5ModelForPrediction
    if _TFB["m"] is None:
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        _TFB["m"] = TimesFm2_5ModelForPrediction.from_pretrained(
            "google/timesfm-2.5-200m-transformers", torch_dtype=torch.bfloat16, device_map=dev).eval()
        _TFB["dev"] = dev
    m, dev = _TFB["m"], _TFB["dev"]
    preds = []
    with torch.no_grad():
        for w in windows:
            ti = torch.tensor(np.asarray(w, np.float32), device=dev).unsqueeze(0)
            preds.append(m(past_values=ti).mean_predictions[0, :hmax].float().cpu().numpy())
    return np.array(preds)

def load_eu_series():
    d = pd.read_csv(SWISS); d["date"] = pd.to_datetime(d["date"])
    out = {}
    for loc, g in d.groupby("location"):
        s = (g.set_index("date")[TARGET].resample("W").mean()
               .interpolate(limit=3).dropna())
        s = s[s > 0]
        if len(s) >= MIN_WEEKS:
            out[f"CH_{loc}"] = np.log10(s.values.astype(float))
    return out

def run(models, tag):
    series = load_eu_series()
    print(f"[eu] cities={len(series)} {list(series)} | weeks {min(len(v) for v in series.values())}-{max(len(v) for v in series.values())}", flush=True)
    rows = []; hmax = max(CFG["horizons"])
    tfm = [m for m in models if m == "timesfm_zs"]   # transformers GPU zero-shot
    seq = [m for m in models if m not in ("timesfm_zs",)]
    for ci, (cid, vals) in enumerate(series.items()):
        n = len(vals); origins = list(range(n - CFG["n_origins"] - hmax, n - hmax))
        for hist in CFG["histories"]:
            for miss in CFG["missing_levels"]:
                for seed in CFG["seeds"]:
                    rng = np.random.default_rng(seed)
                    trains = {}
                    for o in origins:
                        base = vals[max(0, o-hist):o]
                        trains[o] = wf.fast_inject(base, miss, rng) if len(base) >= 10 else None
                    for m in seq:
                        errs = {h: [] for h in CFG["horizons"]}
                        for o in origins:
                            tr = trains[o]
                            if tr is None: continue
                            for h in CFG["horizons"]:
                                errs[h].append(wf.forecast(m, tr, h, seed, key=(cid,hist,miss,seed,o)) - vals[o+h-1])
                        for h, e in errs.items():
                            if e:
                                e=np.array(e); rows.append(dict(region="EU", site=cid, model=m, horizon=h,
                                    history=hist, missing=miss, seed=seed, mae=float(np.mean(np.abs(e))),
                                    rmse=float(math.sqrt(np.mean(e**2))), n_test=len(e)))
                        wf._LSTM_CACHE.clear()
                    for m in tfm:
                        win=[trains[o] for o in origins if trains[o] is not None]
                        oks=[o for o in origins if trains[o] is not None]
                        if not win: continue
                        pf = _tf_zs_predict(win, hmax)
                        for h in CFG["horizons"]:
                            e=np.array([pf[i,h-1]-vals[o+h-1] for i,o in enumerate(oks)])
                            rows.append(dict(region="EU", site=cid, model=m, horizon=h, history=hist,
                                missing=miss, seed=seed, mae=float(np.mean(np.abs(e))),
                                rmse=float(math.sqrt(np.mean(e**2))), n_test=len(e)))
        print(f"[eu city {ci+1}/{len(series)}] {cid}", flush=True)
    df = pd.DataFrame(rows)
    out = os.path.join(wf.DATA_DIR, f"raw_results_eu_{tag}.csv")
    df.to_csv(out, index=False); print(f"[eu] saved {out} rows={len(df)}", flush=True)
    # quick ranking
    rank = df.groupby("model")["mae"].mean().sort_values()
    print("[eu] MAE ranking:", {k: round(v,3) for k,v in rank.items()}, flush=True)
    return df

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default=",".join(wf.BASELINES))
    a = ap.parse_args()
    run(a.models.split(","), tag="_".join(a.models.split(",")) if len(a.models.split(","))<=3 else "multi")
