import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
BASE="/sessions/epic-keen-bardeen/mnt/claude_research/tideguard"
OUT=BASE+"/figure/fig_05_chemical.png"
RED="#C24A45"; SLATE="#5B8DB8"; TEAL="#3E9AA0"; GREY="#9a9a9a"; INK="#2b2b2b"; GRID="#E8E8E8"
plt.rcParams.update({"font.family":"DejaVu Sans","text.color":INK,"axes.edgecolor":"#9a9a9a",
    "axes.labelcolor":INK,"xtick.color":"#444","ytick.color":INK})
FS_T=15; FS_L=13; FS_TK=11.5

SW=BASE+"/data/swiss_wbe/data/04_intermediate/Drugs_Viruses_PopNormLoad.csv"
d=pd.read_csv(SW); d["date"]=pd.to_datetime(d["date"],errors="coerce")
viruses=[c for c in ["SARS-CoV-2","Influenza A virus","Influenza B virus","RSV"] if c in d.columns]
drugs=[c for c in d.columns if c not in ["Unnamed: 0","date","location"]+viruses]
# weekly wide per location, then pooled lead-lag screen
def series(loc,col):
    s=d[d.location==loc].set_index("date")[col].resample("W").mean().interpolate(limit=3)
    return s
def zc(x): 
    x=(x-np.nanmean(x))/(np.nanstd(x)+1e-9); return x
sel_lags=[]; pairs=[]
for loc in d.location.unique():
    for vir in viruses:
        sv=series(loc,vir)
        for dr in drugs:
            sd=series(loc,dr)
            df=pd.concat([sv,sd],axis=1).dropna()
            if len(df)<40: continue
            a=zc(df.iloc[:,0].values); b=zc(df.iloc[:,1].values)
            best=(0,0)
            for lag in range(-6,7):  # drug leads virus if lag>0
                if lag>=0: x=b[:len(b)-lag]; y=a[lag:]
                else: x=b[-lag:]; y=a[:len(a)+lag]
                if len(x)<30: continue
                r=np.corrcoef(x,y)[0,1]
                if abs(r)>abs(best[1]): best=(lag,r)
            if abs(best[1])>0.3:
                sel_lags.append(best[0]); pairs.append((loc,vir,dr,best[0],best[1]))
sel_lags=np.array(sel_lags)

fig=plt.figure(figsize=(16,4.8),dpi=220)
gs=fig.add_gridspec(1,3,wspace=0.40,left=0.06,right=0.985,top=0.82,bottom=0.19)
axa=fig.add_subplot(gs[0,0]); axb=fig.add_subplot(gs[0,1]); axc=fig.add_subplot(gs[0,2])

# (a) per-virus uni vs cov MAE (measured)
import json; pv=json.load(open(BASE+"/data/rq2_summary.json"))["per_virus"]
labs=list(pv.keys()); uni=[pv[k]["uni"] for k in labs]; cov=[pv[k]["cov"] for k in labs]
short={"SARS-CoV-2":"SARS-CoV-2","Influenza A virus":"Influenza A","Influenza B virus":"Influenza B","RSV":"RSV"}
y=np.arange(len(labs)); h=0.36
axa.barh(y+h/2,uni,h,color=SLATE,label="univariate",zorder=3)
axa.barh(y-h/2,cov,h,color=RED,label="+ drug covariates",zorder=3)
axa.set_yticks(y); axa.set_yticklabels([short[k] for k in labs],fontsize=FS_TK)
axa.set_xlabel(r"MAE ($\log_{10}$ units)",fontsize=FS_L)
axa.set_title("Drug covariates raised pooled\nvirus-forecast error",fontsize=FS_T,pad=6)
axa.legend(fontsize=FS_TK-1,frameon=False,loc="lower right")
axa.xaxis.grid(True,color=GRID,lw=1); axa.set_axisbelow(True); axa.tick_params(length=0,labelsize=FS_TK)
for sp in ["top","right","left"]: axa.spines[sp].set_visible(False)
axa.spines["bottom"].set_color("#aaa")

# (b) selected-lag distribution
bins=np.arange(-6.5,7.5,1)
axb.hist(sel_lags,bins=bins,color=TEAL,edgecolor="white",zorder=3)
axb.axvline(0,color=GREY,lw=1.4,ls=(0,(4,3)))
axb.set_xlabel(r"Selected drug$\rightarrow$virus lag (weeks)",fontsize=FS_L)
axb.set_ylabel("Pair count",fontsize=FS_L)
axb.set_title("Exploratory lead–lag screen\n(lag of maximum $|r|$, $|r|>0.3$)",fontsize=FS_T,pad=6)
axb.text(0.03,0.96,f"median = {int(np.median(sel_lags))} wk\n$n$ = {len(sel_lags)} pairs\n(not FDR-controlled)",
         transform=axb.transAxes,va="top",ha="left",fontsize=FS_TK,
         bbox=dict(boxstyle="round,pad=0.35",fc="white",ec="#d9d2c7",lw=1))
axb.grid(True,axis="y",color=GRID,lw=1); axb.set_axisbelow(True); axb.tick_params(length=0,labelsize=FS_TK)
for sp in ["top","right"]: axb.spines[sp].set_visible(False)
for sp in ["left","bottom"]: axb.spines[sp].set_color("#aaa")

# (c) illustrative lag-selected overlay: pick a positive-lead antihistamine-influenza pair
cand=[p for p in pairs if "Influenza" in p[1] and 2<=p[3]<=5 and p[4]>0.3]
pick=max(cand,key=lambda p:abs(p[4])) if cand else max(pairs,key=lambda p:abs(p[4]))
loc,vir,dr,lag,r=pick
sv=zc(series(loc,vir).dropna().values[-90:]); sd=zc(series(loc,dr).dropna().values[-90:])
n=min(len(sv),len(sd)); sv=sv[-n:]; sd=sd[-n:]
axc.plot(np.arange(n),sv,color=RED,lw=2,label=f"{vir.replace(' virus','')}")
axc.plot(np.arange(n),sd,color=SLATE,lw=2,label=f"{dr}")
axc.set_title("Illustrative lag-selected overlay",fontsize=FS_T,pad=6)
axc.set_xlabel("Time (weeks)",fontsize=FS_L); axc.set_ylabel("Standardised load",fontsize=FS_L)
axc.legend(fontsize=FS_TK-1,frameon=False,loc="upper right")
axc.text(0.03,0.04,f"{loc}; selected lag {lag:+d} wk, $r$={r:.2f}\n(exploratory, not validated)",
         transform=axc.transAxes,va="bottom",ha="left",fontsize=FS_TK-1,color="#666",style="italic")
axc.grid(True,color=GRID,lw=1); axc.set_axisbelow(True); axc.tick_params(length=0,labelsize=FS_TK)
for sp in ["top","right"]: axc.spines[sp].set_visible(False)
for sp in ["left","bottom"]: axc.spines[sp].set_color("#aaa")

for ax,lab in [(axa,"a"),(axb,"b"),(axc,"c")]:
    ax.text(-0.12,1.14,lab,transform=ax.transAxes,fontsize=21,fontweight="bold",va="top",ha="left")
fig.savefig(OUT,bbox_inches="tight",facecolor="white")
print("saved",OUT,"| lead-lag pairs=%d median=%d | overlay=%s/%s lag%+d r=%.2f"%(len(sel_lags),int(np.median(sel_lags)),pick[2],pick[1],lag,r))
