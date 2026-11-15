import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
BASE="/sessions/epic-keen-bardeen/mnt/claude_research/tideguard"
OUT=BASE+"/figure/Figure_S_screening.png"
INK="#2b2b2b"; BOX="#F3EEE6"; EDGE="#C9BBA7"; RED="#C24A45"; TAN="#8a7f6d"; EX="#efe7db"
plt.rcParams.update({"font.family":"DejaVu Sans","text.color":INK})
fig,ax=plt.subplots(figsize=(9.2,8.4),dpi=200); ax.set_xlim(0,10); ax.set_ylim(0,12); ax.axis("off")
def box(x,y,w,h,txt,fc=BOX,ec=EDGE,fs=12.5,bold=False):
    ax.add_patch(FancyBboxPatch((x-w/2,y-h/2),w,h,boxstyle="round,pad=0.08,rounding_size=0.12",
        fc=fc,ec=ec,lw=1.6,zorder=2))
    ax.text(x,y,txt,ha="center",va="center",fontsize=fs,color=INK,zorder=3,
            fontweight=("bold" if bold else "normal"))
def arrow(y1,y2,x=5.0):
    ax.add_patch(FancyArrowPatch((x,y1),(x,y2),arrowstyle="-|>",mutation_scale=18,lw=2,color=TAN,zorder=1))
def exc(y,txt):
    ax.add_patch(FancyBboxPatch((7.15,y-0.42),3.4,0.84,boxstyle="round,pad=0.06,rounding_size=0.1",
        fc=EX,ec="#d8cdbb",lw=1.2,zorder=2))
    ax.text(8.85,y,txt,ha="center",va="center",fontsize=10.3,color="#6b6357",style="italic",zorder=3)
    ax.add_patch(FancyArrowPatch((5.0,y),(7.15,y),arrowstyle="-|>",mutation_scale=13,lw=1.4,color="#c3b7a3",zorder=1))
main=[
 (11.1,"CDC NWSS SARS-CoV-2 concentration dataset\n(resource g653-rqe2): 1,187 sampling sites",BOX,False),
 (9.1,"1,052 sites with flow–population normalisation",BOX,False),
 (6.9,"230 sites with ≥156 weekly points (≥3 years)\nafter weekly aggregation and ≤3-week gap interpolation",BOX,False),
 (4.4,"State-diversified selection\n(≤2 sites per state, longest series first)",BOX,False),
 (2.2,"16 US NWSS series across 12 states\n(final forecasting panel)",RED,True),
]
ys=[m[0] for m in main]
box(5,11.1,7.6,1.15,main[0][1]); 
box(5,9.1,6.4,0.95,main[1][1])
box(5,6.9,7.4,1.25,main[2][1])
box(5,4.4,6.2,1.05,main[3][1])
ax.add_patch(FancyBboxPatch((5-3.2,2.2-0.6),6.4,1.2,boxstyle="round,pad=0.08,rounding_size=0.12",fc="#f6e4e2",ec=RED,lw=2,zorder=2))
ax.text(5,2.2,main[4][1],ha="center",va="center",fontsize=13,fontweight="bold",color="#7d2f2c",zorder=3)
for a,b in [(11.1,9.1),(9.1,6.9),(6.9,4.4),(4.4,2.2)]:
    arrow(a-0.62,b+0.62)
exc(10.1,"135 sites: microbial\nnormalisation only → excluded")
exc(8.0,"non-positive / non-detect\nconcentrations dropped before log10")
exc(5.65,"822 sites: < 156 weekly\npoints → excluded")
ax.text(5,11.95,"Site screening for the US NWSS panel",ha="center",fontsize=15.5,fontweight="bold",color=INK)
fig.savefig(OUT,bbox_inches="tight",facecolor="white")
import shutil; shutil.copy(OUT,"/sessions/epic-keen-bardeen/mnt/outputs/Figure_S_screening.png")
print("saved",OUT)
