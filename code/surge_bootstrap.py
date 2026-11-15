#!/usr/bin/env python3
"""Site-clustered bootstrap CIs for surge early-warning metrics (EM&S revision).
Reads data/alert_perorigin.csv (site, model, pred_rise, surge).
Outputs point AUC + det@10%FA with 95% CIs, and paired TimesFM-vs-baseline diffs.
numpy-only (np.trapezoid); no scipy dependency."""
import os, json, numpy as np, pandas as pd
np.random.seed(20260930)
HERE=os.path.dirname(os.path.abspath(__file__))
CSV=os.path.join(HERE,"..","..","data","alert_perorigin.csv")
df=pd.read_csv(CSV)
sites=sorted(df.site.unique()); B=4000
FA=0.10

def roc_from(sub):
    y=sub.surge.values.astype(bool); s=sub.pred_rise.values.astype(float)
    P=y.sum(); N=(~y).sum()
    if P==0 or N==0: return np.nan, np.nan
    ths=np.unique(np.concatenate([[-1e9],np.sort(s),[1e9]]))
    tpr=np.empty(len(ths)); fpr=np.empty(len(ths))
    for k,t in enumerate(ths):
        al=s>=t; tpr[k]=(al&y).sum()/P; fpr[k]=(al&~y).sum()/N
    order=np.argsort(fpr); auc=np.trapezoid(tpr[order],fpr[order])
    idx=np.where(fpr<=FA)[0]; det=tpr[idx].max() if len(idx) else 0.0
    return float(auc), float(det)

# pre-index rows by (site,model) for fast bootstrap
grp={m:{sid:df[(df.model==m)&(df.site==sid)] for sid in sites} for m in df.model.unique()}
def metric_for(model, site_list):
    sub=pd.concat([grp[model][sid] for sid in site_list], ignore_index=True)
    return roc_from(sub)

models=["timesfm","chronos","seasonal_naive","ar_ols","ridge_ar","ets","naive"]
point={m:metric_for(m,sites) for m in models}

# bootstrap: resample sites with replacement
boot={m:{"auc":[],"det":[]} for m in models}
diff={m:{"auc":[],"det":[]} for m in models if m!="timesfm"}
for b in range(B):
    bs=list(np.random.choice(sites,size=len(sites),replace=True))
    mvals={m:metric_for(m,bs) for m in models}
    for m in models:
        boot[m]["auc"].append(mvals[m][0]); boot[m]["det"].append(mvals[m][1])
    for m in models:
        if m=="timesfm": continue
        diff[m]["auc"].append(mvals["timesfm"][0]-mvals[m][0])
        diff[m]["det"].append(mvals["timesfm"][1]-mvals[m][1])

def ci(a): a=np.array(a); return float(np.percentile(a,2.5)), float(np.percentile(a,97.5))
def p_two(a):  # bootstrap two-sided p for diff>0
    a=np.array(a); return float(2*min((a<=0).mean(),(a>=0).mean()))

print(f"{'model':14s}{'AUC':>7s}{'  95% CI':>16s}{'det@10%FA':>11s}{'  95% CI':>16s}")
out={"n_sites":len(sites),"B":B,"tau":0.30,"H":2,"fa":FA,"point":{},"ci":{},"paired_vs_timesfm":{}}
for m in models:
    a=point[m][0]; d=point[m][1]; al,ah=ci(boot[m]["auc"]); dl,dh=ci(boot[m]["det"])
    print(f"{m:14s}{a:7.3f}  [{al:.3f},{ah:.3f}] {d:9.3f}  [{dl:.3f},{dh:.3f}]")
    out["point"][m]={"auc":round(a,3),"det10":round(d,3)}
    out["ci"][m]={"auc":[round(al,3),round(ah,3)],"det10":[round(dl,3),round(dh,3)]}
print("\nPaired diff (TimesFM - baseline):")
for m in diff:
    da=np.mean(diff[m]["auc"]); al,ah=ci(diff[m]["auc"]); pa=p_two(diff[m]["auc"])
    dd=np.mean(diff[m]["det"]); dl,dh=ci(diff[m]["det"]); pd_=p_two(diff[m]["det"])
    print(f"  vs {m:14s} dAUC={da:+.3f} [{al:+.3f},{ah:+.3f}] p={pa:.3f} | ddet={dd:+.3f} [{dl:+.3f},{dh:+.3f}] p={pd_:.3f}")
    out["paired_vs_timesfm"][m]={"dauc":round(da,3),"dauc_ci":[round(al,3),round(ah,3)],"dauc_p":round(pa,3),
                                 "ddet":round(dd,3),"ddet_ci":[round(dl,3),round(dh,3)],"ddet_p":round(pd_,3)}
json.dump(out,open(os.path.join(HERE,"surge_bootstrap_results.json"),"w"),indent=2)
print("\nsaved -> surge_bootstrap_results.json")
