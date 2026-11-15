#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
tideguard (#24) EM&S revision -- LoRA SWEEP (answers R1.4), FAST-EVAL version.

Re-runs the matched-backend LoRA train (finetune_timesfm.py) across a grid of
configurations and, for each, compares the held-out MAE of the fine-tuned model
vs the zero-shot base. Evaluation uses a single full-history (104-week, no-missing)
chronological config over all sites -- enough to answer "does any configuration
beat zero-shot?" without re-running the full histories x missing x seeds sweep
(that full sweep, in finetune_timesfm.evaluate, is what makes a naive sweep take
many hours).

Place next to finetune_timesfm.py (tideguard ROOT). Needs torch + transformers +
peft + accelerate.

Usage (Ubuntu, GPU env 'tideguard'):
    python lora_sweep.py --smoke     # 2 configs, 2 sites (validity, ~minutes)
    python lora_sweep.py             # default grid (6 configs), fast eval
    python lora_sweep.py --full      # full grid (rank x lr x epochs = 24)
Output (data/):
    lora_sweep.csv   rank, alpha, lr, epochs, mae_zs, mae_ft, delta, better
"""
import os, argparse, importlib.util, types
import numpy as np, pandas as pd, torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = HERE if os.path.exists(os.path.join(HERE, "finetune_timesfm.py")) else os.path.abspath(os.path.join(HERE, "..", ".."))
def _imp(name, fn):
    spec = importlib.util.spec_from_file_location(name, os.path.join(ROOT, fn))
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m
ft = _imp("ft", "finetune_timesfm.py")
DATA = ft.DATA_DIR
HIST = 104  # full-history chronological eval config

def cfg(rank, lr, epochs, alpha=None, smoke=False):
    return types.SimpleNamespace(context_len=64, horizon_len=8, epochs=epochs, batch_size=32,
                                 lr=lr, lora_r=rank, lora_alpha=(alpha if alpha else 2*rank),
                                 num_samples=(3000 if smoke else 5000), seed=42,
                                 eval_only=False, smoke=smoke)

def fast_eval(adapter_dir, smoke):
    """Single full-history chronological eval over all sites: mean held-out MAE for
    zero-shot base vs merged fine-tuned model."""
    from transformers import TimesFm2_5ModelForPrediction
    from peft import PeftModel
    dev = ft.device()
    base = TimesFm2_5ModelForPrediction.from_pretrained(ft.MODEL_ID, torch_dtype=torch.bfloat16, device_map=dev).eval()
    ftb = TimesFm2_5ModelForPrediction.from_pretrained(ft.MODEL_ID, torch_dtype=torch.bfloat16, device_map=dev)
    ftm = PeftModel.from_pretrained(ftb, adapter_dir).merge_and_unload().eval()
    series = ft.wf.load_series(); hmax = max(ft.CFG["horizons"])
    sites = list(series.items())[:2] if smoke else list(series.items())
    zs, fm = [], []
    for sid, s in sites:
        v = s.values.astype(float); n = len(v)
        origins = list(range(n - ft.CFG["n_origins"] - hmax, n - hmax))
        wins = [v[max(0, o - HIST):o] for o in origins]
        zp = ft._predict(base, wins, hmax, dev); fp = ft._predict(ftm, wins, hmax, dev)
        for h in ft.CFG["horizons"]:
            yt = np.array([v[o + h - 1] for o in origins])
            zs.append(float(np.mean(np.abs(zp[:, h - 1] - yt))))
            fm.append(float(np.mean(np.abs(fp[:, h - 1] - yt))))
    del base, ftb, ftm
    if dev == "cuda": torch.cuda.empty_cache()
    return float(np.mean(zs)), float(np.mean(fm))

def run_one(a):
    tag = f"r{a.lora_r}_a{a.lora_alpha}_lr{a.lr:g}_e{a.epochs}"
    ft.ADAPTER_DIR = os.path.join(ROOT, f"timesfm_wbe_lora_{tag}")
    ft.train_adapter(a)
    zs, fm = fast_eval(ft.ADAPTER_DIR, a.smoke)
    return dict(rank=a.lora_r, alpha=a.lora_alpha, lr=a.lr, epochs=a.epochs,
                mae_zs=round(zs, 4), mae_ft=round(fm, 4), delta=round(fm - zs, 4), better=bool(fm < zs))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true"); ap.add_argument("--full", action="store_true")
    a = ap.parse_args()
    if a.smoke:
        grid = [cfg(4, 1e-4, 4, smoke=True), cfg(8, 5e-5, 4, smoke=True)]
    elif a.full:
        grid = [cfg(r, lr, e) for r in (2, 4, 8, 16) for lr in (5e-5, 1e-4, 3e-4) for e in (5, 10)]
    else:
        grid = [cfg(4, 1e-4, 10), cfg(8, 1e-4, 10), cfg(16, 1e-4, 10),
                cfg(4, 5e-5, 10), cfg(8, 5e-5, 10), cfg(8, 3e-4, 10)]
    rows = []
    for i, a1 in enumerate(grid):
        print(f"\n########## config {i+1}/{len(grid)}: r{a1.lora_r} a{a1.lora_alpha} lr{a1.lr:g} e{a1.epochs} ##########", flush=True)
        try:
            r = run_one(a1); rows.append(r)
            print(f"   -> zs={r['mae_zs']}  ft={r['mae_ft']}  delta={r['delta']}  better={r['better']}", flush=True)
        except Exception as e:
            print("[config failed]", repr(e), flush=True)
        pd.DataFrame(rows).to_csv(os.path.join(DATA, "lora_sweep.csv"), index=False)
    df = pd.DataFrame(rows)
    print("\n=== LoRA sweep (zero-shot base MAE vs fine-tuned MAE) ===")
    print(df.to_string(index=False))
    if len(df):
        print(f"\nAny configuration beat zero-shot? {'YES' if df.better.any() else 'NO'}")
        print("[saved] data/lora_sweep.csv")

if __name__ == "__main__":
    main()
