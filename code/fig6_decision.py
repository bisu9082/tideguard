import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
BASE="/sessions/epic-keen-bardeen/mnt/claude_research/tideguard"
OUT=BASE+"/figure/Figure_6.png"
df=pd.read_csv(BASE+"/data/alert_perorigin.csv")
RED="#C24A45"; PURPLE="#8E6BB0"; TEAL="#3E9AA0"; ORANGE="#E8943A"; SLATE="#5B8DB8"; GREY="#8a8f96"; INK="#2b2b2b"; GRID="#E8E8E8"
plt.rcParams.update({"font.family":"DejaVu Sans","text.color":INK,"axes.edgecolor":"#9a9a9a","axes.labelcolor":INK,"xtick.color":"#444","ytick.color":INK})
FS_T=18; FS_L=16; FS_TK=13
def roc(m):
    d=df[df.model==m]; y=d.surge.values.astype(bool); s=d.pred_rise.values
    P=y.sum(); N=(~y).sum(); ths=np.unique(np.concatenate([[-1e9],np.sort(s),[1e9]]))
    tpr=np.array([((s>=t)&y).sum()/P for t in ths]); fpr=np.array([((s>=t)&~y).sum()/N for t in ths])
    o=np.argsort(fpr); auc=float(np.trapezoid(tpr[o],fpr[o]))
    idx=np.where(fpr<=0.10)[0]; det10=float(tpr[idx].max()) if len(idx) else 0.0
    return fpr[o],tpr[o],auc,det10
disp={"timesfm":("TimesFM",RED,3.0),"seasonal_naive":("Seasonal-naive",TEAL,1.8),
      "ar_ols":("AR-OLS",ORANGE,1.8),"chronos":("Chronos-Bolt",PURPLE,1.8),
      "ridge_ar":("Ridge-AR",SLATE,1.8),"ets":("ETS","#6E9E5B",1.8),"naive":("Naive (persistence)",GREY,2.6)}
fig=plt.figure(figsize=(14,6),dpi=220)
gs=fig.add_gridspec(1,2,width_ratios=[1.15,1],wspace=0.30,left=0.075,right=0.985,top=0.88,bottom=0.14)
axa=fig.add_subplot(gs[0,0]); axb=fig.add_subplot(gs[0,1])
# ROC
order=["timesfm","seasonal_naive","ar_ols","chronos","ridge_ar","ets","naive"]
dets={}
for m in order:
    fpr,tpr,auc,d10=roc(m); dets[m]=d10; nm,c,lw=disp[m]
    axa.plot(fpr,tpr,color=c,lw=lw,label=f"{nm} (AUC {auc:.2f})",zorder=(5 if m=='timesfm' else 3))
axa.plot([0,1],[0,1],color="#bbb",lw=1.2,ls=(0,(4,3)),zorder=1)
axa.axvline(0.10,color="#ccc",lw=1,ls=":")
axa.set_xlabel("False-alarm rate",fontsize=FS_L); axa.set_ylabel("Surge detection rate",fontsize=FS_L)
axa.set_title("Surge early-warning (ROC)",fontsize=FS_T,pad=8)
axa.set_xlim(0,1); axa.set_ylim(0,1.02); axa.grid(True,color=GRID,lw=1); axa.set_axisbelow(True)
axa.legend(fontsize=FS_TK-1,frameon=False,loc="lower right"); axa.tick_params(length=0,labelsize=FS_TK)
for sp in ["top","right"]: axa.spines[sp].set_visible(False)
for sp in ["left","bottom"]: axa.spines[sp].set_color("#aaa")
# detection@10%FA bar
bo=sorted(order,key=lambda m:dets[m])
labels=[disp[m][0] for m in bo]; vals=[dets[m] for m in bo]
cols=[RED if m=="timesfm" else GREY if m=="naive" else "#BDB2A6" for m in bo]
y=np.arange(len(bo)); axb.barh(y,vals,color=cols,height=0.7,edgecolor="white",zorder=3)
axb.set_yticks(y); axb.set_yticklabels(labels,fontsize=FS_TK)
for yi,v in zip(y,vals): axb.text(v+0.008,yi,f"{v:.2f}",va="center",ha="left",fontsize=FS_TK,color="#333")
axb.set_xlabel("Surge detection at 10% false-alarm",fontsize=FS_L)
axb.set_title("Early-warning value diverges from MAE",fontsize=FS_T,pad=8)
axb.set_xlim(0,0.60); axb.grid(True,axis="x",color=GRID,lw=1); axb.set_axisbelow(True); axb.tick_params(length=0,labelsize=FS_TK)
for sp in ["top","right","left"]: axb.spines[sp].set_visible(False)
axb.spines["bottom"].set_color("#aaa")
axb.text(0.02,0.4,"persistence: lowest MAE,\nzero early warning",transform=axb.transAxes,fontsize=FS_TK-1,style="italic",color=GREY)
for ax,lab,dx in [(axa,"a",-0.10),(axb,"b",-0.22)]:
    ax.text(dx,1.10,lab,transform=ax.transAxes,fontsize=24,fontweight="bold",va="top",ha="left")
fig.suptitle("Decision-focused validation: forecasting adds surge early-warning value that point error hides",
             fontsize=14.5,y=0.985,color="#555")
fig.savefig(OUT,bbox_inches="tight",facecolor="white")
print("saved",OUT)
