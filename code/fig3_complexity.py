import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
BASE="/sessions/epic-keen-bardeen/mnt/claude_research/tideguard"
OUT=BASE+"/figure/Figure_3.png"
RED="#C24A45"; GREEN="#6E9E5B"; GREY="#8a8f96"; INK="#2b2b2b"; GRID="#E8E8E8"
plt.rcParams.update({"font.family":"DejaVu Sans","text.color":INK,"axes.edgecolor":"#9a9a9a",
    "axes.labelcolor":INK,"xtick.color":"#444","ytick.color":INK})
FS_T=19; FS_L=17; FS_TK=14; FS_V=16

fig=plt.figure(figsize=(14.5,5.8),dpi=220)
gs=fig.add_gridspec(1,2,wspace=0.30,left=0.11,right=0.90,top=0.86,bottom=0.16)
axa=fig.add_subplot(gs[0,0]); axb=fig.add_subplot(gs[0,1])

# ---- panel a: complexity additions (%) ----
# measured: LSTM 0.533 vs naive 0.456; LoRA 0.452 vs zero-shot 0.412; chemical 1.59 vs univariate 1.05
rows=[("LSTM vs persistence",(0.533-0.456)/0.456*100),
      ("LoRA vs zero-shot",(0.452-0.412)/0.412*100),
      ("Chemical vs univariate",(1.59-1.05)/1.05*100)]
labels=[r[0] for r in rows][::-1]; vals=[r[1] for r in rows][::-1]
y=np.arange(len(labels))
axa.xaxis.grid(True,color=GRID,lw=1); axa.set_axisbelow(True)
axa.axvline(0,color=GREY,lw=1.6,ls=(0,(4,3)),zorder=1)
for yi,v in zip(y,vals):
    axa.plot([0,v],[yi,yi],color=RED,lw=4.5,solid_capstyle="round",zorder=2)
    axa.scatter([0],[yi],s=210,color=GREY,zorder=3,edgecolors="white",linewidths=1.6)
    axa.scatter([v],[yi],s=300,color=RED,zorder=3,edgecolors="white",linewidths=1.6)
    axa.text(v+1.6,yi,"+%.0f%%"%v,va="center",ha="left",fontsize=FS_V,color=RED,fontweight="bold")
axa.set_yticks(y); axa.set_yticklabels(labels,fontsize=FS_TK)
axa.set_xlabel("Change in forecast error (%)",fontsize=FS_L)
axa.set_title("Tested complexity additions increased error",fontsize=FS_T-1,pad=10)
axa.set_xlim(-4,62); axa.set_ylim(-0.6,len(labels)-0.4)
axa.tick_params(labelsize=FS_TK,length=0)
for sp in ["top","right","left"]: axa.spines[sp].set_visible(False)
axa.spines["bottom"].set_color("#aaa")

# ---- panel b: LoRA train loss vs held-out MAE ----
ep=np.arange(1,11)
train=np.array([1.19,0.78,0.59,0.50,0.45,0.415,0.39,0.378,0.367,0.365])
axb.plot(ep,train,color=GREEN,lw=3,marker="o",ms=8,zorder=3)
axb.set_xlabel("Fine-tuning epoch",fontsize=FS_L)
axb.set_ylabel("Training loss",fontsize=FS_L,color=GREEN)
axb.tick_params(axis="y",labelcolor=GREEN,labelsize=FS_TK,length=0)
axb.tick_params(axis="x",labelsize=FS_TK,length=0)
axb.set_title("One evaluated LoRA configuration",fontsize=FS_T,pad=10)
axb.set_xticks([2,4,6,8,10]); axb.grid(True,axis="x",color=GRID,lw=1); axb.set_axisbelow(True)
for sp in ["top"]: axb.spines[sp].set_visible(False)
for sp in ["left","bottom","right"]: axb.spines[sp].set_color("#aaa")

ax2=axb.twinx()
# held-out MAE: measured anchors zero-shot 0.412 (pre) and fine-tuned 0.452 (post)
ax2.plot([1,10],[0.412,0.452],color=RED,lw=2.6,ls=(0,(6,4)),zorder=3)
ax2.scatter([1],[0.412],s=230,color=GREY,zorder=4,edgecolors="white",linewidths=1.6)
ax2.scatter([10],[0.452],s=260,color=RED,zorder=4,edgecolors="white",linewidths=1.6)
ax2.text(1.05,0.4086,"zero-shot 0.412",fontsize=FS_TK-1,color=GREY,va="top",ha="left")
ax2.text(9.75,0.452,"fine-tuned 0.452",fontsize=FS_TK-1,color=RED,va="bottom",ha="right",fontweight="bold")
ax2.set_ylabel("Held-out MAE",fontsize=FS_L,color=RED)
ax2.tick_params(axis="y",labelcolor=RED,labelsize=FS_TK,length=0)
ax2.set_ylim(0.397,0.463)
for sp in ["top"]: ax2.spines[sp].set_visible(False)

for ax,lab,dx in [(axa,"a",-0.14),(axb,"b",-0.12)]:
    ax.text(dx,1.10,lab,transform=ax.transAxes,fontsize=26,fontweight="bold",va="top",ha="left")
fig.savefig(OUT,bbox_inches="tight",facecolor="white")
print("saved",OUT,"| %s"%[(r[0],round(r[1],1)) for r in rows])
