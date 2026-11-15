import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import matplotlib.colors as mc
BASE="/sessions/epic-keen-bardeen/mnt/claude_research/tideguard"
OUT=BASE+"/figure/graphical_abstract.png"
RED="#C24A45"; TEAL="#3E9AA0"; INK="#2b2b2b"; GREY="#7a7a7a"
def tint(h,a=0.10):
    r,g,b=mc.to_rgb(h); return (r+(1-r)*(1-a),g+(1-g)*(1-a),b+(1-b)*(1-a))
plt.rcParams.update({"font.family":"DejaVu Sans","text.color":INK})
# WR spec ~2.5:1
fig=plt.figure(figsize=(13.75,5.5),dpi=220); ax=fig.add_axes([0,0,1,1]); ax.set_xlim(0,2.5); ax.set_ylim(0,1); ax.axis("off")

ax.text(1.25,0.955,"Pre-trained models help some wastewater signals, not all",
        ha="center",va="center",fontsize=19,fontweight="bold")

rng=np.random.default_rng(3)
def panel(cx,color,title,sig,rough,l1,l2,winner):
    w=0.72
    ax.add_patch(FancyBboxPatch((cx-w/2,0.30),w,0.52,boxstyle="round,pad=0.01,rounding_size=0.03",
        lw=2.4,edgecolor=color,facecolor=tint(color)))
    ax.text(cx,0.775,title,ha="center",va="center",fontsize=13.5,fontweight="bold",color=color)
    ax.text(cx,0.715,r"roughness $\sigma_\Delta$ = %s"%sig,ha="center",va="center",fontsize=12,color=INK)
    # mini trace
    n=90; t=np.linspace(0,1,n)
    if rough: y=np.cumsum(rng.normal(0,1,n)); y=(y-y.mean())/y.std()
    else:
        y=np.sin(t*6)*1.0+np.sin(t*2.3)*0.4; y=(y-y.mean())/y.std()
    xs=cx-0.30+t*0.60; ys=0.50+y*0.055
    ax.plot(xs,ys,color=color,lw=1.8)
    ax.text(cx,0.40,l1,ha="center",va="center",fontsize=11.5,color=INK)
    ax.text(cx,0.355,l2,ha="center",va="center",fontsize=11.5,color=INK)
    ax.text(cx,0.325,winner,ha="center",va="center",fontsize=11,style="italic",color=color,fontweight="bold")

panel(0.62,RED,"United States — rougher","0.72",True,
      "TimesFM  0.412",  "persistence  0.456","foundation model lower error")
panel(1.88,TEAL,"Switzerland — smoother","0.36",False,
      "persistence  0.203","TimesFM  0.214","persistence lower error")

# center: shared benchmark + moderator
ax.add_patch(FancyBboxPatch((1.10,0.36),0.30,0.40,boxstyle="round,pad=0.01,rounding_size=0.03",
    lw=1.6,edgecolor="#b7ae9f",facecolor="white"))
ax.text(1.25,0.72,"Shared benchmark",ha="center",fontsize=11.5,fontweight="bold")
ax.text(1.25,0.685,"11 models",ha="center",fontsize=10.2,color=GREY)
ax.text(1.25,0.658,"16 US + 10 Swiss sites",ha="center",fontsize=10.2,color=GREY)
ax.text(1.25,0.631,"rolling origins",ha="center",fontsize=10.2,color=GREY)
# mini moderator scatter
gd=pd.read_csv(BASE+"/data/site_roughness_gain.csv")
sx=1.135+ (gd.sigma-gd.sigma.min())/(gd.sigma.max()-gd.sigma.min())*0.23
sy=0.44+ (gd.gain-gd.gain.min())/(gd.gain.max()-gd.gain.min())*0.13
cols=[RED if s=="US" else TEAL for s in gd.system]
ax.scatter(sx,sy,s=10,c=cols,zorder=3)
ax.text(1.25,0.415,r"gain vs $\sigma_\Delta$: $r$=0.70",ha="center",fontsize=9.6,color=INK)
# arrows
ax.add_patch(FancyArrowPatch((0.985,0.56),(1.095,0.56),arrowstyle="-|>",mutation_scale=14,lw=1.8,color="#888"))
ax.add_patch(FancyArrowPatch((1.405,0.56),(1.515,0.56),arrowstyle="-|>",mutation_scale=14,lw=1.8,color="#888"))

ax.text(1.25,0.075,"Model rankings differed between the two systems; signal roughness may help identify\nwhen strong local baselines should be prioritised.",
        ha="center",va="center",fontsize=12.5,color=INK)
fig.savefig(OUT,bbox_inches="tight",facecolor="white")
from PIL import Image; im=Image.open(OUT); print("saved",OUT,im.size,"ratio=%.2f"%(im.size[0]/im.size[1]))
