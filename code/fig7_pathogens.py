import pandas as pd, numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
B="/sessions/epic-keen-bardeen/mnt/claude_research/tideguard"
RED="#C24A45"; ORANGE="#E8943A"; GREEN="#6E9E5B"; GREY="#8a8f96"; TAN="#C7B79A"; INK="#2b2b2b"; GRID="#E8E8E8"
plt.rcParams.update({"font.family":"DejaVu Sans","text.color":INK,"axes.edgecolor":"#9a9a9a","axes.labelcolor":INK,"xtick.color":"#444","ytick.color":INK})
def roc(sub):
    y=sub.surge.values.astype(bool); s=sub.pred_rise.values.astype(float); P=y.sum(); N=(~y).sum()
    if P==0 or N==0: return np.nan
    ths=np.unique(np.concatenate([[-1e9],np.sort(s),[1e9]]))
    tpr=np.array([((s>=t)&y).sum()/P for t in ths]); fpr=np.array([((s>=t)&~y).sum()/N for t in ths])
    o=np.argsort(fpr); return float(np.trapezoid(tpr[o],fpr[o]))
def ptt(pc,a,b):
    d=(pc[a]-pc[b]).dropna(); return (float(d.mean()),float(d.mean()/(d.std(ddof=1)/np.sqrt(len(d)))),len(d))
# --- SARS matched (hist=104, miss=0) ---
bl=pd.read_csv(B+"/data/raw_results_baselines.csv"); tz=pd.read_csv(B+"/data/raw_results_timesfm_zs.csv")
def matched(df): 
    d=df[(df.history==104)&(df.missing==0.0)]; return d.groupby(["site","model"]).mae.mean().unstack()
pc_s=matched(bl).join(matched(tz).rename(columns={"timesfm_zs":"timesfm"})[["timesfm"]],how="outer")
sars_mae=pc_s.mean()
al_s=pd.read_csv(B+"/data/alert_perorigin.csv")
sars_auc={m:roc(al_s[al_s.model==m]) for m in ["timesfm","ar_ols","ridge_ar","naive","seasonal_naive"]}
# --- 3 pathogens ---
mae=pd.read_csv(B+"/data/raw_results_pathogens.csv"); al=pd.read_csv(B+"/data/alert_perorigin_pathogens.csv")
PATHS=[("sars","SARS-CoV-2\n(n=16)"),("fluB","Influenza B\n(n=10)"),("rsv","RSV\n(n=8)"),("fluA","Influenza A\n(n=4)")]
MODELS=[("timesfm","TimesFM",RED),("ar_ols","AR-OLS",ORANGE),("ridge_ar","Ridge-AR",GREEN),
        ("naive","Persistence",GREY),("seasonal_naive","Seasonal-naive",TAN)]
maeval={"sars":sars_mae}; aucval={"sars":sars_auc}
for tag in ["fluB","rsv","fluA"]:
    maeval[tag]=mae[mae.pathogen==tag].groupby("model").mae.mean()
    aucval[tag]={m:roc(al[(al.pathogen==tag)&(al.model==m)]) for m,_,_ in MODELS}
# paired t SARS timesfm vs ar_ols
print("SARS matched paired TimesFM-AR_OLS:",[round(v,4) for v in ptt(pc_s,"timesfm","ar_ols")])
fig,(axa,axb)=plt.subplots(1,2,figsize=(15.5,5.8),dpi=220)
fig.subplots_adjust(left=0.065,right=0.985,top=0.80,bottom=0.13,wspace=0.20)
fig.suptitle("Across four pathogens in two surveillance systems, the foundation model beats only the trivial baselines,\nnot a simple autoregression — and the point-accuracy leader (persistence) never issues an early warning",
             fontsize=13,y=0.965,color="#444")
x=np.arange(len(PATHS)); w=0.16
for j,(mk,nm,c) in enumerate(MODELS):
    axa.bar(x+(j-2)*w,[maeval[t].get(mk,np.nan) for t,_ in PATHS],w,color=c,label=nm,edgecolor="white",linewidth=0.6,zorder=3)
axa.set_xticks(x); axa.set_xticklabels([l for _,l in PATHS],fontsize=12); axa.set_ylabel(r"Mean MAE ($\log_{10}$ units)",fontsize=13.5)
axa.set_title("Point-forecast error (full-history config)",fontsize=15,pad=8); axa.grid(True,axis="y",color=GRID,lw=1); axa.set_axisbelow(True)
axa.tick_params(length=0,labelsize=11.5); axa.legend(fontsize=10.5,frameon=False,ncol=2,loc="upper left")
for sp in ["top","right"]: axa.spines[sp].set_visible(False)
for sp in ["left","bottom"]: axa.spines[sp].set_color("#aaa")
for j,(mk,nm,c) in enumerate(MODELS):
    axb.bar(x+(j-2)*w,[aucval[t][mk] for t,_ in PATHS],w,color=c,edgecolor="white",linewidth=0.6,zorder=3)
axb.axhline(0.5,color="#b9b0a2",lw=1.3,ls=(0,(4,3)),zorder=2); axb.text(3.4,0.505,"chance",fontsize=10,color="#8a8f96",style="italic",va="bottom",ha="right")
axb.set_xticks(x); axb.set_xticklabels([l for _,l in PATHS],fontsize=12); axb.set_ylabel("Surge-detection AUC",fontsize=13.5)
axb.set_title("Decision-focused early warning",fontsize=15,pad=8); axb.set_ylim(0.45,0.90); axb.grid(True,axis="y",color=GRID,lw=1); axb.set_axisbelow(True)
axb.tick_params(length=0,labelsize=11.5)
for sp in ["top","right"]: axb.spines[sp].set_visible(False)
for sp in ["left","bottom"]: axb.spines[sp].set_color("#aaa")
for ax,lab in [(axa,"a"),(axb,"b")]: ax.text(-0.085,1.10,lab,transform=ax.transAxes,fontsize=22,fontweight="bold",va="top",ha="left")
fig.savefig(B+"/figure/Figure_7.png",bbox_inches="tight",facecolor="white")
import shutil; shutil.copy(B+"/figure/Figure_7.png","/sessions/epic-keen-bardeen/mnt/outputs/Figure_7_pathogens.png")
for t,_ in PATHS: print(t,"MAE",{m:round(float(maeval[t].get(m,np.nan)),3) for m,_,_ in MODELS},"AUC",{m:round(aucval[t][m],3) for m,_,_ in MODELS})
print("saved")
