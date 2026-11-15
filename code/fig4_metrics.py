import numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
BASE="/sessions/epic-keen-bardeen/mnt/claude_research/tideguard"
OUT=BASE+"/figure/Figure_4.png"
INK="#2b2b2b"; GRID="#E8E8E8"; GREENBOX="#6E9E5B"
# (MAE, dir_acc, color, marker, size)
P={
 "TimesFM":      (0.412,0.624,"#C24A45","*",620),
 "Chronos-Bolt": (0.437,0.596,"#8E6BB0","o",260),
 "Ridge-AR":     (0.458,0.437,"#5B8DB8","o",240),
 "ETS":          (0.498,0.545,"#3E9AA0","o",240),
 "AR-OLS":       (0.598,0.583,"#E8943A","o",240),
 "Persistence":  (0.456,0.000,"#7F7F7F","o",260),
 "Seasonal-naive":(0.507,0.159,"#B0B0B0","o",240),
}
plt.rcParams.update({"font.family":"DejaVu Sans","text.color":INK,"axes.edgecolor":"#9a9a9a",
    "axes.labelcolor":INK,"xtick.color":"#444","ytick.color":INK})
fig,ax=plt.subplots(figsize=(11.5,8.6),dpi=220)
ax.grid(True,color=GRID,lw=1); ax.set_axisbelow(True)
# green "sweet spot" box
ax.add_patch(FancyBboxPatch((0.401,0.565),0.052,0.092,boxstyle="round,pad=0.002,rounding_size=0.006",
    lw=0,facecolor=GREENBOX,alpha=0.14,zorder=1))
ax.text(0.427,0.685,"low error +\nhigh directional accuracy",ha="center",va="center",
        fontsize=14,color=GREENBOX,fontweight="bold",zorder=5)
# random reference
ax.axhline(0.5,color="#9a9a9a",lw=1.6,ls=(0,(6,4)),zorder=1)
ax.text(0.612,0.512,"random-call reference (0.5)",ha="right",va="bottom",fontsize=13,color="#7a7a7a")
# points
for name,(x,y,c,mk,s) in P.items():
    ax.scatter([x],[y],s=s,color=c,marker=mk,zorder=4,edgecolors="white",linewidths=1.6)
# labels (manual offsets, no overlap)
lab={"TimesFM":(0.417,0.636,"left","bottom","#C24A45",True),
     "Chronos-Bolt":(0.443,0.578,"left","top","#8E6BB0",True),
     "Ridge-AR":(0.463,0.437,"left","center","#5B8DB8",False),
     "ETS":(0.503,0.552,"left","bottom","#3E9AA0",False),
     "AR-OLS":(0.593,0.596,"right","bottom","#E8943A",False),
     "Persistence":(0.456,0.052,"center","bottom","#5a5a5a",False),
     "Seasonal-naive":(0.513,0.163,"left","center","#8a8a8a",False)}
for name,(x,y,ha,va,c,bold) in lab.items():
    ax.text(x,y,name,ha=ha,va=va,fontsize=14,color=c,fontweight=("bold" if bold else "normal"),zorder=6)
# annotation to persistence
ax.annotate("no directional call\n(call coverage = 0)",xy=(0.456,0.000),xytext=(0.505,0.085),
    fontsize=12.5,color="#5a5a5a",ha="left",va="bottom",
    arrowprops=dict(arrowstyle="-|>",color="#8a8a8a",lw=1.8,mutation_scale=22,
                    shrinkB=15,shrinkA=4))
ax.set_xlabel(r"Point error, MAE ($\log_{10}$ units; lower is better)",fontsize=17,labelpad=8)
ax.set_ylabel("Directional accuracy (higher is better)",fontsize=17,labelpad=8)
ax.set_title("Point error and directional behaviour provide different information",
             fontsize=18,pad=12,fontweight="bold")
ax.set_xlim(0.398,0.618); ax.set_ylim(-0.035,0.72)
ax.tick_params(labelsize=14,length=0)
for sp in ["top","right"]: ax.spines[sp].set_visible(False)
for sp in ["left","bottom"]: ax.spines[sp].set_color("#aaa")
fig.text(0.5,0.005,"Persistence issues no directional call; its y=0 position is a plotting convention, not classifier accuracy.",
         ha="center",fontsize=12,style="italic",color="#6a6a6a")
fig.savefig(OUT,bbox_inches="tight",facecolor="white")
print("saved",OUT)
