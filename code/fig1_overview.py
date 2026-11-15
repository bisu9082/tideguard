import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
BASE="/sessions/epic-keen-bardeen/mnt/claude_research/tideguard"
OUT=BASE+"/figure/Figure_1.png"
INK="#2b2b2b"
# stage accent colors (muted)
C_DATA="#5B8DB8"; C_PROC="#CE9A46"; C_MODEL="#3E9AA0"; C_EVAL="#6E9E5B"; RQC="#C24A45"
def tint(hexc,a=0.12):
    import matplotlib.colors as mc, numpy as np
    r,g,b=mc.to_rgb(hexc); return (r+(1-r)*(1-a), g+(1-g)*(1-a), b+(1-b)*(1-a))
plt.rcParams.update({"font.family":"DejaVu Sans","text.color":INK})
fig=plt.figure(figsize=(16,9),dpi=200); ax=fig.add_axes([0,0,1,1]); ax.set_xlim(0,16); ax.set_ylim(0,9); ax.axis("off")

def box(cx,cy,w,h,text,edge,fs=13,bold=False,face=None,align="center"):
    if face is None: face=tint(edge)
    ax.add_patch(FancyBboxPatch((cx-w/2,cy-h/2),w,h,boxstyle="round,pad=0.02,rounding_size=0.14",
        linewidth=2.0,edgecolor=edge,facecolor=face,mutation_aspect=1))
    ax.text(cx,cy,text,ha=align,va="center",fontsize=fs,color=INK,
            fontweight=("bold" if bold else "normal"),linespacing=1.35)

def header(cx,txt,col):
    ax.text(cx,8.55,txt,ha="center",va="center",fontsize=17,fontweight="bold",color=col,
            bbox=dict(boxstyle="round,pad=0.3",fc=tint(col,0.18),ec="none"))

cols=[2.55,6.35,10.15,13.95]
header(cols[0],"DATA",C_DATA); header(cols[1],"PROCESSING",C_PROC)
header(cols[2],"MODEL LADDER",C_MODEL); header(cols[3],"EVALUATION",C_EVAL)

# DATA
box(cols[0],6.55,3.0,1.7,"US NWSS\n16 sites | 12 states\nSARS-CoV-2",C_DATA,13,True)
box(cols[0],3.95,3.0,1.7,"Switzerland\n10 cities\n15 drugs | 4 viruses",C_DATA,13,True)
# PROCESSING
box(cols[1],6.55,3.2,1.7,"Within-site weekly series\nlog$_{10}$ transform\nrolling origins: 1, 2, 4 weeks",C_PROC,12.5)
box(cols[1],3.95,3.2,1.7,"Stress tests\n12-104 wk history\n0 / 10 / 30% missing",C_PROC,12.5)
# MODEL LADDER (5 rungs)
rungs=[("Persistence | seasonal-naive","#8a8f96"),
       ("ETS | AR-OLS | Ridge-AR | SARIMA",C_DATA),
       ("XGBoost | LSTM",C_MODEL),
       ("TimesFM zero-shot | TimesFM + LoRA",RQC),
       ("Chronos-Bolt zero-shot","#8E6BB0")]
ys=[7.35,6.15,4.95,3.75,2.55]
for (t,c),y in zip(rungs,ys):
    box(cols[2],y,3.5,1.0,t,c,12,False)
# EVALUATION
box(cols[3],4.95,3.0,3.2,"MAE + site-clustered CI\npaired site-level tests\n\ndirectional behaviour\ninterval coverage",C_EVAL,12.5,True)

# arrows between stages (mid height ~5.0)
def arrow(x0,x1,y):
    ax.add_patch(FancyArrowPatch((x0,y),(x1,y),arrowstyle="-|>",mutation_scale=22,
        lw=2.4,color="#6b6b6b"))
arrow(4.15,4.75,5.0); arrow(7.95,8.55,5.0); arrow(11.95,12.35,5.0)

# RQ row
ax.add_patch(FancyBboxPatch((0.7,0.35),14.6,1.35,boxstyle="round,pad=0.02,rounding_size=0.1",
    lw=0,facecolor="#faf3f2"))
rqs=[("RQ1","When does pre-training\nadd forecast skill?"),
     ("RQ2","Do pharmaceuticals add\nforecast information?"),
     ("RQ3","Does the ranking\nreplicate across systems?")]
rx=[3.4,8.0,12.6]
for (tag,q),x in zip(rqs,rx):
    box(x,1.02,4.2,1.05,"",RQC,face="white")
    ax.text(x-1.75,1.02,tag,ha="left",va="center",fontsize=15,fontweight="bold",color=RQC)
    ax.text(x+0.35,1.02,q,ha="center",va="center",fontsize=12,color=INK,linespacing=1.3)

fig.savefig(OUT,bbox_inches="tight",facecolor="white")
print("saved",OUT)
