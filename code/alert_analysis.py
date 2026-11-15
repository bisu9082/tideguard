#!/usr/bin/env python3
# Decision-focused validation: surge early-warning (detection vs false-alarm)
# Surge = >= TAU log10 rise over horizon H. A model alerts if predicted rise >= threshold.
# Baselines run in sandbox; TimesFM/Chronos need GPU (use --full).
import os, json, argparse, importlib.util
import numpy as np, pandas as pd
import warnings; warnings.filterwarnings("ignore")
HERE=os.path.dirname(os.path.abspath(__file__))
def imp(n,f):
    s=importlib.util.spec_from_file_location(n,os.path.join(HERE,f)); m=importlib.util.module_from_spec(s); s.loader.exec_module(m); return m
wf=imp("wf","wbe_forecast.py"); CFG=wf.CFG
TAU=0.30; H=2; HIST=104
BASE={"naive":wf.f_naive,"ets":wf.f_ets,"ar_ols":wf.f_ar_ols,"ridge_ar":wf.f_ridge_ar,"seasonal_naive":wf.f_seasonal_naive}

def collect(full=False):
    ser=wf.load_series(); rows=[]
    cr=imp("cr","chronos_run.py") if full else None
    for sid,w in ser.items():
        v=np.asarray(w.values,float); n=len(v)
        origins=[o for o in range(n-CFG["n_origins"]-4, n-4) if o-1>=10 and o+H-1<n]
        wins=[v[max(0,o-HIST):o] for o in origins]
        cur=np.array([v[o-1] for o in origins]); tgt=np.array([v[o+H-1] for o in origins])
        surge=(tgt-cur)>=TAU
        for name,fn in BASE.items():
            pr=np.array([fn(wins[i],H)-cur[i] for i in range(len(wins))])
            for i in range(len(wins)): rows.append((sid,name,float(pr[i]),bool(surge[i])))
        if full:
            tf=cr.timesfm_predict(wins,max(CFG["horizons"]))[0][:,H-1]
            ch=cr.chronos_predict(wins,max(CFG["horizons"]))[0][:,H-1]
            for i in range(len(wins)):
                rows.append((sid,"timesfm",float(tf[i]-cur[i]),bool(surge[i])))
                rows.append((sid,"chronos",float(ch[i]-cur[i]),bool(surge[i])))
    return pd.DataFrame(rows,columns=["site","model","pred_rise","surge"])

def roc(df,model):
    d=df[df.model==model]; y=d.surge.values; s=d.pred_rise.values
    P=y.sum(); N=(~y).sum()
    ths=np.unique(np.concatenate([[-1e9],np.sort(s),[1e9]]))
    tpr=[]; fpr=[]
    for t in ths:
        al=s>=t; tpr.append((al&y).sum()/P); fpr.append((al&~y).sum()/N)
    tpr=np.array(tpr); fpr=np.array(fpr)
    order=np.argsort(fpr); auc=np.trapz(tpr[order],fpr[order])
    # detection at 10% false-alarm
    idx=np.where(fpr<=0.10)[0]; det10=tpr[idx].max() if len(idx) else 0.0
    return auc,det10,P,N

if __name__=="__main__":
    ap=argparse.ArgumentParser(); ap.add_argument("--full",action="store_true"); a=ap.parse_args()
    df=collect(a.full); df.to_csv(f"{HERE}/data/alert_perorigin.csv",index=False)
    print(f"surge def: >= {TAU} log10 rise over {H}wk | rows={len(df)}")
    models=["naive","seasonal_naive","ets","ar_ols","ridge_ar"]+(["chronos","timesfm"] if a.full else [])
    print(f"{'model':14s} {'AUC':>6s} {'det@10%FA':>10s}")
    out={}
    for m in models:
        auc,det10,P,N=roc(df,m); out[m]=dict(auc=round(auc,3),det10=round(det10,3))
        print(f"{m:14s} {auc:6.3f} {det10:10.3f}")
    json.dump(out,open(f"{HERE}/data/alert_summary.json","w"),indent=2)
