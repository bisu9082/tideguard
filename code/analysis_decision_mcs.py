# Decision-metric Model Confidence Set (IJF revision lever)
import pandas as pd, numpy as np
from arch.bootstrap import MCS
al=pd.read_csv("data/alert_perorigin.csv")
models=["timesfm","seasonal_naive","ar_ols","ridge_ar","chronos","ets","naive"]
disp={"timesfm":"TimesFM","seasonal_naive":"Seasonal-naive","ar_ols":"AR-OLS","ridge_ar":"Ridge-AR","chronos":"Chronos-Bolt","ets":"ETS","naive":"Naive"}
FA=0.10
def thr(d):
    neg=d.loc[~d.surge,"pred_rise"].values
    return np.quantile(neg,1-FA) if len(neg) else np.inf
sites=sorted(al.site.unique()); M=pd.DataFrame(index=sites)
for m in models:
    d=al[al.model==m]; t=thr(d); per=[]
    for s in sites:
        ds=d[d.site==s]; y=ds.surge.values.astype(bool); a=ds.pred_rise.values>=t
        per.append(((y&~a).sum()+(~y&a).sum())/len(ds))   # misclassification at 10% FA
    M[disp[m]]=per
mcs=MCS(M,size=0.10,reps=10000,block_size=1,method="R",seed=7); mcs.compute()
print("decision 90% MCS INCLUDED:",list(mcs.included))
print("mean loss:",{k:round(v,3) for k,v in M.mean().sort_values().items()})
