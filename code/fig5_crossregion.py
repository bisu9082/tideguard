import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
import scipy.stats as st
BASE="/sessions/epic-keen-bardeen/mnt/claude_research/tideguard"
OUT=BASE+"/figure/Figure_5.png"
RED="#C24A45"; GREY="#7F7F7F"; SLATE="#9DB1C4"; TEAL="#3E9AA0"; INK="#2b2b2b"; GRID="#E8E8E8"
plt.rcParams.update({"font.family":"DejaVu Sans","text.color":INK,"axes.edgecolor":"#9a9a9a",
    "axes.labelcolor":INK,"xtick.color":"#444","ytick.color":INK})
FS_T=17; FS_L=15; FS_TK=13; FS_V=13

us=[("TimesFM",0.412),("Naive",0.456),("Ridge-AR",0.458),("ETS",0.498),("Seas-naive",0.507),("AR-OLS",0.598)]
sw=pd.read_csv(BASE+"/data/raw_results_swiss10.csv")
m=sw.groupby(["city","model"]).mae.mean().unstack().mean()
swiss=sorted([("Naive",float(m["naive"])),("TimesFM",float(m["timesfm"])),("Ridge-AR",float(m["ridge_ar"])),
       ("Seas-naive",float(m["seasonal_naive"])),("ETS",float(m["ets"])),("AR-OLS",float(m["ar_ols"]))],key=lambda x:x[1])
gd=pd.read_csv(BASE+"/data/site_roughness_gain.csv")

fig=plt.figure(figsize=(16.5,5.6),dpi=220)
gs=fig.add_gridspec(1,3,width_ratios=[1,1,1.05],wspace=0.40,left=0.065,right=0.985,top=0.80,bottom=0.17)
axa=fig.add_subplot(gs[0,0]); axb=fig.add_subplot(gs[0,1]); axc=fig.add_subplot(gs[0,2])
fig.suptitle("Point-error ranking differs across systems, and roughness tracks the foundation-model gain",
             fontsize=17.5,fontweight="bold",y=0.965)

def bars(ax,data,title,xmax):
    labels=[n for n,_ in data][::-1]; vals=[v for _,v in data][::-1]; y=np.arange(len(labels))
    cols=[RED if n=="TimesFM" else GREY if n=="Naive" else SLATE for n in labels]
    ax.xaxis.grid(True,color=GRID,lw=1); ax.set_axisbelow(True)
    ax.barh(y,vals,color=cols,height=0.7,edgecolor="white",linewidth=0.8,zorder=3)
    ax.set_yticks(y); ax.set_yticklabels(labels,fontsize=FS_TK)
    ax.set_xlabel(r"MAE ($\log_{10}$ units)",fontsize=FS_L); ax.set_title(title,fontsize=FS_T,pad=8)
    ax.set_xlim(0,xmax)
    for yi,v in zip(y,vals): ax.text(v+xmax*0.02,yi,"%.3f"%v,va="center",ha="left",fontsize=FS_V,color="#333")
    ax.tick_params(length=0,labelsize=FS_TK)
    for sp in ["top","right","left"]: ax.spines[sp].set_visible(False)
    ax.spines["bottom"].set_color("#aaa")
bars(axa,us,"United States (NWSS)",0.70)
bars(axb,swiss,"Switzerland",0.315)

# panel c: site-level moderator scatter
axc.axhline(0,color="#bbb",lw=1.2,zorder=1)
for sysn,c in [("US",RED),("Swiss",TEAL)]:
    g=gd[gd.system==sysn]
    axc.scatter(g.sigma,g.gain,s=95,color=c,edgecolors="white",linewidths=1.2,zorder=3,label=sysn)
# OLS fit all 26
b1,b0=np.polyfit(gd.sigma,gd.gain,1); xs=np.linspace(gd.sigma.min(),gd.sigma.max(),50)
axc.plot(xs,b0+b1*xs,color="#555",lw=2,ls=(0,(5,3)),zorder=2)
r,p=st.pearsonr(gd.sigma,gd.gain)
axc.text(0.03,0.96,f"$r$ = {r:.2f}, $p$ < 0.001\n($n$ = 26 sites)",transform=axc.transAxes,
         va="top",ha="left",fontsize=FS_V+1,color="#333",
         bbox=dict(boxstyle="round,pad=0.4",fc="white",ec="#d9d2c7",lw=1.1))
axc.set_xlabel(r"Signal roughness $\sigma_\Delta$",fontsize=FS_L)
axc.set_ylabel("Foundation-model gain\n(naive $-$ TimesFM MAE)",fontsize=FS_L-1)
axc.set_title("Roughness vs foundation-model gain",fontsize=FS_T,pad=8)
axc.grid(True,color=GRID,lw=1); axc.set_axisbelow(True); axc.tick_params(length=0,labelsize=FS_TK)
axc.legend(loc="lower right",fontsize=FS_TK,frameon=False,handletextpad=0.2)
for sp in ["top","right"]: axc.spines[sp].set_visible(False)
for sp in ["left","bottom"]: axc.spines[sp].set_color("#aaa")

for ax,lab,dx in [(axa,"a",-0.10),(axb,"b",-0.10),(axc,"c",-0.13)]:
    ax.text(dx,1.12,lab,transform=ax.transAxes,fontsize=23,fontweight="bold",va="top",ha="left")
fig.savefig(OUT,bbox_inches="tight",facecolor="white")
print("saved",OUT,"r=%.3f p=%.2g"%(r,p))
