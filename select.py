#!/usr/bin/env python3
"""tideguard selection procedure (demo).

Given per-site point-error losses and per-origin surge-alert outcomes, this script
runs the decision-aware model-selection rule from the paper on the deposited US panel:
  1. point-error Model Confidence Set (MCS; Hansen et al., 2011)
  2. decision-metric MCS (misclassification cost at a 10% false-alarm operating point)
  3. baseline-first recommendation.

Usage:  python select.py
Reproduces the US example from data/derived/. Swap in your own site-level losses
(sites x models) and per-origin alerts (site, model, pred_rise, surge) to apply it
to a new surveillance programme.
"""
import numpy as np, pandas as pd
from arch.bootstrap import MCS

FA = 0.10  # false-alarm operating point

def point_mcs(loss_df, size=0.10, reps=5000):
    m = MCS(loss_df, size=size, reps=reps, block_size=1, method="R", seed=0); m.compute()
    return list(m.included)

def decision_loss(alerts, cost_miss=1.0, cost_fa=1.0):
    sites = sorted(alerts.site.unique()); M = pd.DataFrame(index=sites)
    for model in alerts.model.unique():
        d = alerts[alerts.model == model]
        neg = d.loc[~d.surge, "pred_rise"].values
        thr = np.quantile(neg, 1 - FA) if len(neg) else np.inf
        per = []
        for s in sites:
            ds = d[d.site == s]; y = ds.surge.values.astype(bool); a = ds.pred_rise.values >= thr
            per.append((cost_miss*(y & ~a).sum() + cost_fa*(~y & a).sum()) / len(ds))
        M[model] = per
    return M

if __name__ == "__main__":
    pt = pd.read_csv("data/derived/results_site_level_US.csv").set_index("site")
    pt = pt.rename(columns={"timesfm_zs": "TimesFM"})
    pmcs = point_mcs(pt)
    al = pd.read_csv("data/derived/alert_perorigin.csv")
    dmcs = point_mcs(decision_loss(al))
    print("Point-error 90% MCS :", pmcs)
    print("Decision 90% MCS    :", dmcs)
    best_point = pt.mean().idxmin()
    print(f"\nLowest average error : {best_point} (MAE {pt.mean().min():.3f})")
    if "naive" in [c.lower() for c in dmcs] or "Naive" in dmcs:
        pass
    # baseline-first rule
    simple = {"Naive","naive","AR-OLS","ar_ols","Seasonal-naive","seasonal_naive","Ridge-AR","ridge_ar","ETS","ets"}
    complex_in_decision = [m for m in dmcs if m not in simple]
    if len(pmcs) == 1 and complex_in_decision and all(m in dmcs for m in pmcs):
        verdict = ("A complex model leads on point error but the decision set is wider: "
                   "adopt it ONLY if it beats a transparent baseline on your alert horizon "
                   "and false-alarm tolerance; otherwise keep the baseline (baseline-first).")
    else:
        verdict = "Decision ranking is unresolved across models: retain the transparent baseline (baseline-first)."
    print("\nRECOMMENDATION:", verdict)
