import json, numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
BASE="/sessions/epic-keen-bardeen/mnt/claude_research/tideguard"
OUT=BASE+"/figure/Figure_6.png"
df=pd.read_csv(BASE+"/data/alert_perorigin.csv")
bs=json.load(open(BASE+"/emss/analysis/surge_bootstrap_results.json"))
RED="#C24A45"; PURPLE="#8E6BB0"; TEAL="#3E9AA0"; ORANGE="#E8943A"; GREEN="#6E9E5B"; TAN="#BDB2A6"; GREY="#8a8f96"; INK="#2b2b2b"; GRID="#E8E8E8"
plt.rcParams.update({"font.family":"DejaVu Sans","text.color":INK,"axes.edgecolor":"#9a9a9a","axes.labelcolor":INK,"xtick.color":"#444","ytick.color":INK})
FS_T=17; FS_L=15; FS_TK=13
def roc(m):
    d=df[df.model==m]; y=d.surge.values.astype(bool); s=d.pred_rise.values
    P=y.sum(); N=(~y).sum(); ths=np.unique(np.concatenate([[-1e9],np.sort(s),[1e9]]))
    tpr=np.array([((s>=t)&y).sum()/P for t in ths]); fpr=np.array([((s>=t)&~y).sum()/N for t in ths])
    o=np.argsort(fpr); return fpr[o],tpr[o]
disp={"timesfm":("TimesFM",RED,3.0),"seasonal_naive":("Seasonal-naive",TEAL,1.8),
      "ar_ols":("AR-OLS",ORANGE,1.8),"chronos":("Chronos-Bolt",PURPLE,1.8),
      "ridge_ar":("Ridge-AR",GREEN,1.8),"ets":("ETS",TAN,1.8),"naive":("Naive (persistence)",GREY,2.4)}
order=["timesfm","seasonal_naive","ar_ols","chronos","ridge_ar","ets","naive"]
fig=plt.figure(figsize=(14.5,6),dpi=220)
gs=fig.add_gridspec(1,2,width_ratios=[1.1,1],wspace=0.32,left=0.075,right=0.985,top=0.85,bottom=0.14)
axa=fig.add_subplot(gs[0,0]); axb=fig.add_subplot(gs[0,1])
# --- panel a: ROC ---
for m in order:
    fpr,tpr=roc(m); nm,c,lw=disp[m]; auc=bs["point"][m]["auc"]
    axa.plot(fpr,tpr,color=c,lw=lw,label=f"{nm} (AUC {auc:.2f})",zorder=(5 if m=='timesfm' else 3))
axa.plot([0,1],[0,1],color="#bbb",lw=1.2,ls=(0,(4,3)),zorder=1)
axa.axvline(0.10,color="#ccc",lw=1,ls=":")
axa.set_xlabel("False-alarm rate",fontsize=FS_L); axa.set_ylabel("Surge detection rate",fontsize=FS_L)
axa.set_title("Surge early-warning (ROC)",fontsize=FS_T,pad=8)
axa.set_xlim(0,1); axa.set_ylim(0,1.02); axa.grid(True,color=GRID,lw=1); axa.set_axisbelow(True)
axa.legend(fontsize=FS_TK-1,frameon=False,loc="lower right"); axa.tick_params(length=0,labelsize=FS_TK)
for sp in ["top","right"]: axa.spines[sp].set_visible(False)
for sp in ["left","bottom"]: axa.spines[sp].set_color("#aaa")
# --- panel b: AUC forest plot with 95% CI (site-clustered bootstrap) ---
fo=sorted(order,key=lambda m:bs["point"][m]["auc"])
y=np.arange(len(fo))
for yi,m in zip(y,fo):
    a=bs["point"][m]["auc"]; lo,hi=bs["ci"][m]["auc"]; nm,c,_=disp[m]
    col=RED if m=="timesfm" else GREY if m=="naive" else "#7d7469"
    axb.plot([lo,hi],[yi,yi],color=col,lw=2.4,zorder=2,solid_capstyle="round")
    axb.scatter([a],[yi],s=70,color=col,edgecolors="white",linewidths=1.1,zorder=3)
    axb.text(hi+0.012,yi,f"{a:.2f}",va="center",ha="left",fontsize=FS_TK-1,color="#333")
axb.axvline(0.5,color="#c9c2b6",lw=1.2,ls=(0,(4,3)),zorder=1)
# mark n.s. vs TimesFM for the two overlapping baselines
for m,txt in [("seasonal_naive","n.s. vs TimesFM"),("ar_ols","n.s.")]:
    yi=list(fo).index(m); axb.text(0.505,yi+0.32,txt,fontsize=FS_TK-2,style="italic",color="#8a8f96")
axb.set_yticks(y); axb.set_yticklabels([disp[m][0] for m in fo],fontsize=FS_TK)
axb.set_xlabel("Surge-detection AUC (95% CI, site-clustered)",fontsize=FS_L)
axb.set_title("Foundation model $\\approx$ seasonal baseline",fontsize=FS_T,pad=8)
axb.set_xlim(0.45,0.90); axb.grid(True,axis="x",color=GRID,lw=1); axb.set_axisbelow(True); axb.tick_params(length=0,labelsize=FS_TK)
for sp in ["top","right","left"]: axb.spines[sp].set_visible(False)
axb.spines["bottom"].set_color("#aaa")
for ax,lab,dx in [(axa,"a",-0.10),(axb,"b",-0.24)]:
    ax.text(dx,1.10,lab,transform=ax.transAxes,fontsize=23,fontweight="bold",va="top",ha="left")
fig.suptitle("Model ranking depends on the operational metric: the point-accuracy leader is not the early-warning leader,\nand the foundation model shows no significant surge-warning gain over a simple seasonal baseline",
             fontsize=12.8,y=0.975,color="#555")
fig.savefig(OUT,bbox_inches="tight",facecolor="white")
import shutil; shutil.copy(OUT,"/sessions/epic-keen-bardeen/mnt/outputs/Figure_6_preview.png")
print("saved",OUT)
