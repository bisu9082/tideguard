import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
BASE="/sessions/epic-keen-bardeen/mnt/claude_research/tideguard"
OUT=BASE+"/figure/Figure_S_roughness_split.png"
gd=pd.read_csv(BASE+"/data/site_roughness_gain.csv")
RED="#C24A45"; GREY="#8a8f96"; TEAL="#3E9AA0"; INK="#2b2b2b"; GRID="#E8E8E8"
plt.rcParams.update({"font.family":"DejaVu Sans","text.color":INK,"axes.edgecolor":"#9a9a9a","axes.labelcolor":INK,"xtick.color":"#444","ytick.color":INK})
FS_T=16; FS_L=14; FS_TK=12
rng=np.random.default_rng(7)
def r_p(x,y,B=20000):
    x=np.asarray(x,float); y=np.asarray(y,float); r=np.corrcoef(x,y)[0,1]
    cnt=0
    for _ in range(B):
        if abs(np.corrcoef(x,rng.permutation(y))[0,1])>=abs(r): cnt+=1
    return r,(cnt+1)/(B+1)
sig=gd.sigma.values
rn,pn=r_p(sig,gd.naive.values); rt,pt=r_p(sig,gd.tfm.values); rgn,pgn=r_p(sig,gd.gain.values)
fig,axes=plt.subplots(1,2,figsize=(12,5.2),dpi=220)
fig.subplots_adjust(left=0.08,right=0.985,top=0.82,bottom=0.14,wspace=0.28)
fig.suptitle("Signal roughness raises persistence error faster than foundation-model error",fontsize=16.5,fontweight="bold",y=0.965)
def panel(ax,yv,r,p,color,ylab,title):
    for sysn,mk in [("US","o"),("Swiss","s")]:
        g=gd[gd.system==sysn]; yy=g[yv].values
        ax.scatter(g.sigma,yy,s=80,color=color,marker=mk,edgecolors="white",linewidths=1.1,zorder=3,label=sysn)
    b1,b0=np.polyfit(sig,gd[yv].values,1); xs=np.linspace(sig.min(),sig.max(),40)
    ax.plot(xs,b0+b1*xs,color="#555",lw=2,ls=(0,(5,3)),zorder=2)
    ax.text(0.03,0.96,f"$r$ = {r:.2f}, $p$ = {p:.3f}\n($n$ = 26)",transform=ax.transAxes,va="top",ha="left",
            fontsize=FS_TK+1,color="#333",bbox=dict(boxstyle="round,pad=0.4",fc="white",ec="#d9d2c7",lw=1.1))
    ax.set_xlabel(r"Signal roughness $\sigma_\Delta$",fontsize=FS_L); ax.set_ylabel(ylab,fontsize=FS_L)
    ax.set_title(title,fontsize=FS_T,pad=8); ax.grid(True,color=GRID,lw=1); ax.set_axisbelow(True)
    ax.tick_params(length=0,labelsize=FS_TK); ax.legend(loc="lower right",fontsize=FS_TK,frameon=False)
    for sp in ["top","right"]: ax.spines[sp].set_visible(False)
    for sp in ["left","bottom"]: ax.spines[sp].set_color("#aaa")
panel(axes[0],"naive",rn,pn,GREY,r"Persistence MAE ($\log_{10}$)","Roughness vs persistence error")
panel(axes[1],"tfm",rt,pt,RED,r"TimesFM MAE ($\log_{10}$)","Roughness vs foundation-model error")
for ax,lab in [(axes[0],"a"),(axes[1],"b")]:
    ax.text(-0.13,1.10,lab,transform=ax.transAxes,fontsize=22,fontweight="bold",va="top",ha="left")
fig.savefig(OUT,bbox_inches="tight",facecolor="white")
import shutil; shutil.copy(OUT,"/sessions/epic-keen-bardeen/mnt/outputs/Figure_S_roughness_split.png")
print(f"naive: r={rn:.3f} p={pn:.4f} | tfm: r={rt:.3f} p={pt:.4f} | gain: r={rgn:.3f} p={pgn:.4f}")
print("saved",OUT)
