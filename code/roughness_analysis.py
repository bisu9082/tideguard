import numpy as np, pandas as pd, importlib.util, os, json, collections
HERE="/sessions/epic-keen-bardeen/mnt/claude_research/tideguard"
spec=importlib.util.spec_from_file_location("wf",os.path.join(HERE,"wbe_forecast.py")); wf=importlib.util.module_from_spec(spec); spec.loader.exec_module(wf)
CFG=wf.CFG
def sig(v): z=(v-v.mean())/v.std(); return float(np.std(np.diff(z)))

# ---- US: per-site sigma + per-site MAE (timesfm, naive) ----
ser=wf.load_series()  # 16 chosen sites
D=HERE+"/data"
us=pd.concat([pd.read_csv(f"{D}/raw_results_baselines.csv"),
              pd.read_csv(f"{D}/raw_results_timesfm_zs.csv")],ignore_index=True)
usm=us.groupby(["site","model"]).mae.mean().unstack()
rowsUS=[]
for sid,w in ser.items():
    if sid in usm.index and "timesfm_zs" in usm.columns and "naive" in usm.columns:
        rowsUS.append(dict(system="US",site=sid,sigma=sig(w.values),
                           tfm=usm.loc[sid,"timesfm_zs"],naive=usm.loc[sid,"naive"]))
# ---- Swiss ----
SW=HERE+"/data/swiss_wbe/data/04_intermediate/Drugs_Viruses_PopNormLoad.csv"
sd=pd.read_csv(SW); sd["date"]=pd.to_datetime(sd["date"],errors="coerce")
sw=pd.read_csv(D+"/raw_results_swiss10.csv"); swm=sw.groupby(["city","model"]).mae.mean().unstack()
rowsCH=[]
for loc,g in sd.groupby("location"):
    s=g.set_index("date")["SARS-CoV-2"].resample("W").mean().interpolate(limit=3).dropna(); s=s[s>0]
    if len(s)>=52 and loc in swm.index:
        v=np.log10(s.values.astype(float))
        rowsCH.append(dict(system="Swiss",site=loc,sigma=sig(v),tfm=swm.loc[loc,"timesfm"],naive=swm.loc[loc,"naive"]))
df=pd.DataFrame(rowsUS+rowsCH); df["gain"]=df["naive"]-df["tfm"]  # >0 => TimesFM better

# ---- C1: sigma summaries ----
for sysname,g in df.groupby("system"):
    print(f"[C1] {sysname}: sigma mean={g.sigma.mean():.3f} median={g.sigma.median():.3f} range[{g.sigma.min():.3f},{g.sigma.max():.3f}] n={len(g)}")
# ---- C2: moderator test (gain vs sigma across 26 sites) ----
from math import sqrt
def pearson(x,y):
    x=np.array(x);y=np.array(y); r=np.corrcoef(x,y)[0,1]; n=len(x)
    t=r*sqrt((n-2)/(1-r**2)); return r,t,n
def spearman(x,y):
    import scipy.stats as st
    return st.spearmanr(x,y)
r,t,n=pearson(df.sigma,df.gain)
print(f"[C2] ALL n={n}: Pearson r={r:.3f}, t({n-2})={t:.2f}")
try:
    import scipy.stats as st
    rs,ps=st.spearmanr(df.sigma,df.gain); 
    from scipy.stats import t as tdist
    p_pear=2*tdist.sf(abs(t),n-2)
    print(f"[C2] Pearson p={p_pear:.4g}; Spearman rho={rs:.3f} p={ps:.4g}")
    # OLS slope
    b1,b0=np.polyfit(df.sigma,df.gain,1); print(f"[C2] OLS gain = {b0:.3f} + {b1:.3f}*sigma")
except Exception as e: print("scipy?",e)
df.to_csv(HERE+"/data/site_roughness_gain.csv",index=False)
print("saved data/site_roughness_gain.csv"); print(df.round(3).to_string(index=False))
