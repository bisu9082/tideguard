#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
tideguard (#24) — TimesFM 2.5 LoRA fine-tuning on WBE (CDC NWSS) data.
Based on Google's official example (timesfm-forecasting/examples/finetuning).

Backend note: fine-tuning uses the HuggingFace Transformers checkpoint
(google/timesfm-2.5-200m-transformers) + PEFT LoRA. To compare the fine-tune
EFFECT cleanly (same backend), this script evaluates BOTH:
  - timesfm_tf_zs : transformers base, zero-shot (matched baseline for ft)
  - timesfm_ft    : transformers base + LoRA adapter (fine-tuned)
Leakage control: LoRA is trained ONLY on each series' pre-test portion
(everything before the rolling-origin test window).

Requires (after GPU torch): pip install transformers accelerate peft
Downloads the -transformers checkpoint (~925MB) on first run.

Usage:
  python finetune_timesfm.py --smoke      # quick validity check (1 epoch, 2 sites)
  python finetune_timesfm.py              # full train + eval -> raw_results_timesfm_ft.csv
  python finetune_timesfm.py --eval_only  # reuse saved adapter, just evaluate
"""
import argparse, os, math, importlib.util
import numpy as np, pandas as pd, torch

HERE = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location("wf", os.path.join(HERE, "wbe_forecast.py"))
wf = importlib.util.module_from_spec(spec); spec.loader.exec_module(wf)
CFG = wf.CFG
MODEL_ID = "google/timesfm-2.5-200m-transformers"
ADAPTER_DIR = os.path.join(HERE, "timesfm_wbe_lora")
DATA_DIR = wf.DATA_DIR

def device():
    return "cuda" if torch.cuda.is_available() else "cpu"

# ---- data ----
def pretest_series():
    """Pre-test portion of each series (no leakage) as list of np arrays."""
    series = wf.load_series()
    hmax = max(CFG["horizons"]); tail = CFG["n_origins"] + hmax
    out = []
    for sid, s in series.items():
        v = s.values.astype(float)
        out.append(v[: len(v) - tail])
    return series, out

class RandomWindowDS(torch.utils.data.Dataset):
    def __init__(self, series_list, ctx, hor, num, seed):
        self.s, self.ctx, self.hor, self.samples = series_list, ctx, hor, []
        rng = np.random.default_rng(seed); mn = ctx + hor
        valid = [i for i, x in enumerate(series_list) if len(x) >= mn]
        if not valid:
            raise ValueError("no series long enough for ctx+hor")
        for _ in range(num):
            i = rng.choice(valid); x = series_list[i]
            st = int(rng.integers(0, len(x) - mn + 1))
            self.samples.append((i, st))
    def __len__(self): return len(self.samples)
    def __getitem__(self, k):
        i, st = self.samples[k]; x = self.s[i]
        c = torch.tensor(x[st:st+self.ctx], dtype=torch.float32)
        t = torch.tensor(x[st+self.ctx:st+self.ctx+self.hor], dtype=torch.float32)
        return c, t

# ---- train ----
def train_adapter(args):
    from peft import LoraConfig, get_peft_model
    from transformers import TimesFm2_5ModelForPrediction
    dev = device(); print("[ft] device:", dev, flush=True)
    model = TimesFm2_5ModelForPrediction.from_pretrained(
        MODEL_ID, torch_dtype=torch.bfloat16, device_map=dev)
    ctx = min(args.context_len, model.config.context_length)
    lc = LoraConfig(r=args.lora_r, lora_alpha=args.lora_alpha,
                    target_modules="all-linear", lora_dropout=0.05, bias="none")
    model = get_peft_model(model, lc); model.print_trainable_parameters()
    _, series = pretest_series()
    ds = RandomWindowDS(series, ctx, args.horizon_len, args.num_samples, args.seed)
    dl = torch.utils.data.DataLoader(ds, batch_size=args.batch_size, shuffle=True, drop_last=True)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(args.epochs*len(dl), 1))
    for ep in range(1, args.epochs+1):
        model.train(); tot = 0.0; nb = 0
        for c, t in dl:
            c = c.to(dev); t = t.to(dev)
            out = model(past_values=c, future_values=t, forecast_context_len=ctx)
            out.loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); opt.zero_grad(); sch.step()
            tot += float(out.loss.item()); nb += 1
        print(f"[ft] epoch {ep}/{args.epochs} loss={tot/max(nb,1):.4f}", flush=True)
    os.makedirs(ADAPTER_DIR, exist_ok=True)
    model.save_pretrained(ADAPTER_DIR)
    print("[ft] adapter saved ->", ADAPTER_DIR, flush=True)

# ---- evaluate (base zero-shot + ft), same rolling-origin protocol ----
def _predict(model, windows, hmax, dev):
    preds = []
    with torch.no_grad():
        for w in windows:
            ti = torch.tensor(np.asarray(w, np.float32), device=dev).unsqueeze(0)
            out = model(past_values=ti)
            preds.append(out.mean_predictions[0, :hmax].float().cpu().numpy())
    return np.array(preds)

def evaluate(args):
    from peft import PeftModel
    from transformers import TimesFm2_5ModelForPrediction
    dev = device()
    base = TimesFm2_5ModelForPrediction.from_pretrained(
        MODEL_ID, torch_dtype=torch.bfloat16, device_map=dev); base.eval()
    ft_base = TimesFm2_5ModelForPrediction.from_pretrained(
        MODEL_ID, torch_dtype=torch.bfloat16, device_map=dev)
    ft = PeftModel.from_pretrained(ft_base, ADAPTER_DIR)
    ft = ft.merge_and_unload()  # bake LoRA into weights so its effect is not lost
    ft.eval()
    series = wf.load_series(); hmax = max(CFG["horizons"])
    sites = list(series.items())
    if args.smoke: sites = sites[:2]
    out_csv = os.path.join(DATA_DIR, "raw_results_timesfm_ft.csv")
    if os.path.exists(out_csv) and not args.smoke: os.remove(out_csv)
    first = True
    for si, (sid, s) in enumerate(sites):
        vals = s.values.astype(float); n = len(vals); rows = []
        origins = list(range(n-CFG["n_origins"]-hmax, n-hmax))
        for hist in CFG["histories"]:
            for miss in CFG["missing_levels"]:
                for seed in CFG["seeds"]:
                    rng = np.random.default_rng(seed)
                    wins, oks = [], []
                    for o in origins:
                        bw = vals[max(0, o-hist):o]
                        if len(bw) < 10: continue
                        wins.append(wf.fast_inject(bw, miss, rng)); oks.append(o)
                    if not wins: continue
                    # base = pristine instance, ft = LoRA merged into separate instance
                    for name, mdl in [("timesfm_tf_zs", base), ("timesfm_ft", ft)]:
                        pf = _predict(mdl, wins, hmax, dev)
                        for h in CFG["horizons"]:
                            e = np.array([pf[i, h-1] - vals[o+h-1] for i, o in enumerate(oks)])
                            rows.append(dict(site=sid, state=wf._state(sid), model=name, horizon=h,
                                             history=hist, missing=miss, seed=seed,
                                             mae=float(np.mean(np.abs(e))),
                                             rmse=float(math.sqrt(np.mean(e**2))), n_test=len(e)))
        pd.DataFrame(rows).to_csv(out_csv, mode='a', header=first, index=False); first = False
        print(f"[ft-eval {si+1}/{len(sites)}] {sid[:30]}", flush=True)
    print("[ft] eval done ->", out_csv, flush=True)

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--context_len", type=int, default=64)   # multiple of 32
    p.add_argument("--horizon_len", type=int, default=8)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--lora_r", type=int, default=4)
    p.add_argument("--lora_alpha", type=int, default=8)
    p.add_argument("--num_samples", type=int, default=5000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--eval_only", action="store_true")
    p.add_argument("--smoke", action="store_true", help="1 epoch, few samples, 2 sites")
    a = p.parse_args()
    if a.smoke:
        a.epochs = 4; a.num_samples = 3000
    if not a.eval_only:
        train_adapter(a)
    if os.path.isdir(ADAPTER_DIR):
        evaluate(a)
    else:
        print("[ft] no adapter dir; run training first.")

if __name__ == "__main__":
    main()
