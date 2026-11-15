import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
BASE="/sessions/epic-keen-bardeen/mnt/claude_research/tideguard"
D=BASE+"/data"; OUT=BASE+"/figure/Figure_2.png"
rng=np.random.default_rng(0)
files=["raw_results_baselines","raw_results_xgboost_sarima","raw_results_lstm","raw_results_timesfm_zs"]
df=pd.concat([pd.read_csv(f"{D}/{f}.csv") for f in files],ignore_index=True)

order=["timesfm_zs","sarima","naive","ridge_ar","xgboost","ets","seasonal_naive","lstm","ar_ols"]
disp={"timesfm_zs":"TimesFM","sarima":"SARIMA","naive":"Naive","ridge_ar":"Ridge-AR",
      "xgboost":"XGBoost","ets":"ETS","seasonal_naive":"Seas-naive","lstm":"LSTM","ar_ols":"AR-OLS"}
col={"Naive":"#7F7F7F","Seas-naive":"#5B8DB8","ETS":"#3E9AA0","AR-OLS":"#E8943A",
     "Ridge-AR":"#A9772E","SARIMA":"#6E9E5B","XGBoost":"#8E6BB0","LSTM":"#E7A9A5","TimesFM":"#C24A45"}
INK="#2b2b2b"; GRID="#E8E8E8"
plt.rcParams.update({"font.family":"DejaVu Sans","text.color":INK,"axes.edgecolor":"#9a9a9a",
    "axes.labelcolor":INK,"xtick.color":"#444","ytick.color":INK})

# ---- panel a: point + site-clustered bootstrap CI ----
def site_ci(m):
    d=df[df.model==m]
    persite=d.groupby("site").mae.mean().values
    pt=persite.mean()
    boot=[np.mean(rng.choice(persite,len(persite),replace=True)) for _ in range(2000)]
    return pt,np.percentile(boot,2.5),np.percentile(boot,97.5)
rowsA=[(disp[m],*site_ci(m)) for m in order]

# ---- panels b/c/d aggregations ----
def curve(dim):
    g=df[df.model.isin(order)].groupby(["model",dim]).mae.mean().reset_index()
    return g
gb=curve("history"); gc=curve("missing"); gc["missing"]=gc["missing"]*100; gd=curve("horizon")

fig=plt.figure(figsize=(15,10.5),dpi=200)
gs=fig.add_gridspec(2,2,hspace=0.34,wspace=0.28,left=0.11,right=0.83,top=0.93,bottom=0.08)
axa=fig.add_subplot(gs[0,0]); axb=fig.add_subplot(gs[0,1]); axc=fig.add_subplot(gs[1,0]); axd=fig.add_subplot(gs[1,1])

FS_T=19; FS_L=17; FS_TK=14
# panel a
labels=[r[0] for r in rowsA][::-1]; pts=[r[1] for r in rowsA][::-1]
lo=[r[2] for r in rowsA][::-1]; hi=[r[3] for r in rowsA][::-1]
y=np.arange(len(labels))
axa.xaxis.grid(True,color=GRID,lw=1); axa.set_axisbelow(True)
for yi,(lab,p,l,h) in enumerate(zip(labels,pts,lo,hi)):
    c=col[lab]; win=lab=="TimesFM"
    axa.errorbar(p,yi,xerr=[[p-l],[h-p]],fmt="o",color=c,ms=(13 if win else 9),
                 elinewidth=2.2,capsize=5,capthick=2.2,zorder=3,
                 markeredgecolor="white",markeredgewidth=1.3)
axa.set_yticks(y); axa.set_yticklabels(labels,fontsize=FS_TK)
for t,lab in zip(axa.get_yticklabels(),labels):
    if lab=="TimesFM": t.set_color(col["TimesFM"]); t.set_fontweight("bold")
axa.set_xlabel(r"MAE ($\log_{10}$ conc.)",fontsize=FS_L)
axa.set_title("Forecast error (site-clustered 95% CI)",fontsize=FS_T,pad=8)
axa.tick_params(labelsize=FS_TK,length=0)
for sp in ["top","right","left"]: axa.spines[sp].set_visible(False)
axa.spines["bottom"].set_color("#aaa")

def lineplot(ax,g,dim,xlab,title,xticks):
    for m in order:
        dn=disp[m]; sub=g[g.model==m].sort_values(dim)
        win=dn=="TimesFM"
        ax.plot(sub[dim],sub.mae,marker="o",color=col[dn],lw=(3.4 if win else 1.8),
                ms=(9 if win else 6),zorder=(5 if win else 3),label=dn)
    ax.set_xlabel(xlab,fontsize=FS_L); ax.set_ylabel(r"MAE ($\log_{10}$)",fontsize=FS_L)
    ax.set_title(title,fontsize=FS_T,pad=8); ax.set_xticks(xticks)
    ax.grid(True,color=GRID,lw=1); ax.set_axisbelow(True); ax.tick_params(labelsize=FS_TK,length=0)
    for sp in ["top","right"]: ax.spines[sp].set_visible(False)
    for sp in ["left","bottom"]: ax.spines[sp].set_color("#aaa")
lineplot(axb,gb,"history","Training history (weeks)","Data sparsity",[12,26,52,104])
lineplot(axc,gc,"missing","Injected missingness (%)","Robustness to gaps",[0,10,30])
lineplot(axd,gd,"horizon","Forecast horizon (weeks)","Horizon",[1,2,4])

# shared legend (right of figure)
h,l=axb.get_legend_handles_labels()
fig.legend(h,l,loc="center left",bbox_to_anchor=(0.845,0.5),fontsize=FS_TK,frameon=False,
           handlelength=1.8,title="Model",title_fontsize=FS_TK+1)
for ax,lab in [(axa,"a"),(axb,"b"),(axc,"c"),(axd,"d")]:
    ax.text(-0.13,1.13,lab,transform=ax.transAxes,fontsize=26,fontweight="bold",va="top",ha="left")
fig.savefig(OUT,bbox_inches="tight",facecolor="white")
print("saved",OUT)
print("panel a points:",[(r[0],round(r[1],3),round(r[2],3),round(r[3],3)) for r in rowsA])
