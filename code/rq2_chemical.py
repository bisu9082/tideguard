#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
tideguard (#24) RQ2 — does co-located CHEMICAL (drug) signal improve PATHOGEN
forecasting? (cross-causal / leading-indicator test)

Data: Swiss co-located drugs+viruses (ERIC 10.25678/000D6F),
      swiss_wbe/data/04_intermediate/Drugs_Viruses_PopNormLoad.csv
      (date x location, 15 drugs + 4 viruses, population-normalized loads).

Test: forecast each virus h-steps ahead with ridge-AR, comparing
  (A) univariate  : virus own lags only
  (B) +covariate  : virus lags + current+lagged drug loads
A significant MAE reduction (B<A) => chemical carries predictive info beyond
the pathogen's own history => RQ2 supported.
Pure numpy/pandas (no torch). Also reports drug->virus lead-lag cross-correlation.

Out: data/rq2_results.csv, data/rq2_summary.json
"""
import os, json, math, itertools
import numpy as np, pandas as pd
warn = __import__("warnings"); warn.filterwarnings("ignore")

HERE = os.path.dirname(os.path.abspath(__file__))
CSV = os.path.join(HERE, "data", "swiss_wbe", "data", "04_intermediate",
                   "Drugs_Viruses_PopNormLoad.csv")
OUT_CSV = os.path.join(HERE, "data", "rq2_results.csv")
OUT_JSON = os.path.join(HERE, "data", "rq2_summary.json")

VIRUSES = ["SARS-CoV-2", "Influenza A virus", "Influenza B virus", "RSV"]
# respiratory-correlated drugs highlighted by the source study (leading indicators)
DRUGS = ["Acetaminophen", "Codeine", "Dextrorphan", "Pheniramine",
         "N-desmethylpheniramine", "Clarithromycin"]

CFG = dict(freq="W", lags=4, drug_lags=2, horizons=[1, 2], n_origins=20,
           min_points=60, ridge=1.0, seeds=[0, 1, 2, 3, 4], bootstrap=2000)

# ---------- helpers ----------
def _ridge_fit(X, y, lam):
    Xd = np.hstack([np.ones((len(X), 1)), X])
    A = Xd.T @ Xd + lam * np.eye(Xd.shape[1])
    return np.linalg.solve(A, Xd.T @ y)

def _design(varr, darr, t, lags, dlags, use_drug):
    """Feature row to predict index t from info up to t-1."""
    feats = list(varr[t-lags:t][::-1])                 # virus lags 1..lags
    if use_drug and darr is not None:
        for j in range(darr.shape[1]):
            col = darr[:, j]
            feats += list(col[t-dlags:t][::-1])         # drug lags 1..dlags (incl. t-1)
    return np.array(feats, float)

def _smape(y, p):
    d = np.abs(y)+np.abs(p); d[d==0]=1e-9; return 100*np.mean(2*np.abs(p-y)/d)

def bootstrap_ci(x, n, rng, a=0.05):
    x=np.asarray(x,float)
    if len(x)==0: return (float("nan"),)*2
    bs=[np.mean(rng.choice(x,len(x),replace=True)) for _ in range(n)]
    return float(np.percentile(bs,100*a/2)), float(np.percentile(bs,100*(1-a/2)))

def cohens_d(a,b):
    a,b=np.asarray(a,float),np.asarray(b,float)
    if len(a)<2 or len(b)<2: return float("nan")
    sp=math.sqrt(((len(a)-1)*a.var(ddof=1)+(len(b)-1)*b.var(ddof=1))/(len(a)+len(b)-2))
    return float((a.mean()-b.mean())/sp) if sp>0 else float("nan")

def dm(e1,e2,h=1):
    e1,e2=np.asarray(e1,float),np.asarray(e2,float); d=e1**2-e2**2; n=len(d)
    if n<8: return (float("nan"),float("nan"))
    db=d.mean(); var=np.sum((d-db)**2)/n
    if var<=0: return (float("nan"),float("nan"))
    s=db/math.sqrt(var/n); return float(s), float(2*(1-0.5*(1+math.erf(abs(s)/math.sqrt(2)))))

def lead_lag(drug, vir, maxlag=8):
    """Cross-correlation; positive lag => drug LEADS virus."""
    best=(0,0.0)
    for L in range(0, maxlag+1):
        if L==0: a,b=drug,vir
        else: a,b=drug[:-L],vir[L:]
        m=np.isfinite(a)&np.isfinite(b)
        if m.sum()>10:
            c=np.corrcoef(a[m],b[m])[0,1]
            if abs(c)>abs(best[1]): best=(L,float(c))
    return best

# ---------- main experiment ----------
def build_loc(df, loc):
    g=df[df.location==loc].set_index("date").sort_index()
    g=g[~g.index.duplicated()]
    w=g.resample(CFG["freq"]).mean(numeric_only=True)
    # log-transform loads (positive), drugs forward-filled up to 2 weeks
    return w

def run():
    df=pd.read_csv(CSV); df["date"]=pd.to_datetime(df["date"])
    rows=[]; ll_rows=[]
    for loc in sorted(df.location.unique()):
        w=build_loc(df, loc)
        for vir in VIRUSES:
            if vir not in w.columns: continue
            v=w[vir].astype(float)
            # drugs present at this loc
            dcols=[d for d in DRUGS if d in w.columns and w[d].notna().sum()>=CFG["min_points"]]
            if v.notna().sum()<CFG["min_points"] or not dcols: continue
            # align: weekly grid where virus present; ffill drugs up to 2 wks
            sub=w[[vir]+dcols].copy()
            sub[dcols]=sub[dcols].ffill(limit=2)
            sub=sub.dropna(subset=[vir]+dcols)
            if len(sub)<CFG["min_points"]: continue
            varr=np.log10(np.clip(sub[vir].values,1e-9,None))
            darr=np.log10(np.clip(sub[dcols].values,1e-9,None))
            n=len(varr)
            # lead-lag (best drug)
            for j,dn in enumerate(dcols):
                L,c=lead_lag(darr[:,j],varr); ll_rows.append(dict(location=loc,virus=vir,drug=dn,lead_weeks=L,corr=round(c,3)))
            lags,dlags=CFG["lags"],CFG["drug_lags"]
            start=max(lags,dlags)
            for h in CFG["horizons"]:
                origins=range(max(start, n-CFG["n_origins"]-h), n-h)
                for seed in CFG["seeds"]:
                    rng=np.random.default_rng(seed)
                    for use_drug in (False, True):
                        errs=[]
                        for o in origins:
                            # train on [start, o), predict o+h-1 (use info up to o)
                            Xtr=[]; ytr=[]
                            for t in range(start, o):
                                if t+ (h-1) < o:
                                    Xtr.append(_design(varr,darr,t,lags,dlags,use_drug)); ytr.append(varr[t+h-1])
                            if len(Xtr)<8: continue
                            beta=_ridge_fit(np.array(Xtr),np.array(ytr),CFG["ridge"])
                            xq=_design(varr,darr,o,lags,dlags,use_drug)
                            pred=float(beta[0]+np.dot(beta[1:],xq))
                            errs.append(pred-varr[o+h-1])
                        if len(errs)<5: continue
                        e=np.array(errs)
                        rows.append(dict(location=loc,virus=vir,model=("cov" if use_drug else "uni"),
                                         horizon=h,seed=seed,n_drug=len(dcols),
                                         mae=float(np.mean(np.abs(e))),rmse=float(math.sqrt(np.mean(e**2))),n_test=len(e)))
    res=pd.DataFrame(rows); res.to_csv(OUT_CSV,index=False)
    ll=pd.DataFrame(ll_rows)
    print("[rq2] rows:",len(res),"| loc x virus combos:",res.groupby(['location','virus']).ngroups)
    # paired uni vs cov
    rng=np.random.default_rng(7)
    piv=res.pivot_table(index=["location","virus","horizon","seed"],columns="model",values="mae")
    pair=piv.dropna()
    summ=dict(config=CFG,n_pairs=int(len(pair)))
    summ["mae_uni"]=float(pair["uni"].mean()); summ["mae_cov"]=float(pair["cov"].mean())
    summ["improvement_pct"]=float(100*(pair["uni"].mean()-pair["cov"].mean())/pair["uni"].mean())
    summ["uni_ci"]=bootstrap_ci(pair["uni"].values,CFG["bootstrap"],rng)
    summ["cov_ci"]=bootstrap_ci(pair["cov"].values,CFG["bootstrap"],rng)
    rp=res.pivot_table(index=["location","virus","horizon","seed"],columns="model",values="rmse").dropna()
    summ["dm_cov_vs_uni"]=list(dm(rp["uni"].values,rp["cov"].values))  # >0 & p<.05 => cov better
    summ["cohens_d"]=cohens_d(pair["uni"].values,pair["cov"].values)
    # per-virus
    summ["per_virus"]={}
    for vir in VIRUSES:
        pv=res[res.virus==vir].pivot_table(index=["location","horizon","seed"],columns="model",values="mae").dropna()
        if len(pv): summ["per_virus"][vir]=dict(uni=round(float(pv["uni"].mean()),4),cov=round(float(pv["cov"].mean()),4),n=int(len(pv)))
    # lead-lag summary (median lead weeks where corr>0.3)
    if len(ll):
        strong=ll[ll["corr"].abs()>0.3]
        summ["leadlag_median_weeks"]=float(strong.lead_weeks.median()) if len(strong) else None
        summ["leadlag_examples"]=strong.sort_values("corr",ascending=False).head(8).to_dict("records")
    json.dump(summ,open(OUT_JSON,"w"),indent=2,default=str)
    print("[rq2] uni MAE=%.4f  cov MAE=%.4f  improvement=%.1f%%"%(summ["mae_uni"],summ["mae_cov"],summ["improvement_pct"]))
    print("[rq2] DM(uni vs cov) stat,p =",[round(x,4) for x in summ["dm_cov_vs_uni"]],"| cohen d=%.3f"%summ["cohens_d"])
    print("[rq2] per-virus:",summ["per_virus"])
    print("[rq2] lead-lag median weeks (|r|>0.3):",summ.get("leadlag_median_weeks"))
    return summ

if __name__=="__main__":
    run()
