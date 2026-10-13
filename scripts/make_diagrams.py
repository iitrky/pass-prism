"""Draw Fig. 1 (system model) and Fig. 2 (PRISM block diagram)."""
import os, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT); os.chdir(ROOT)
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Circle, Rectangle, FancyArrowPatch
plt.rcParams.update({'font.family':'serif','font.serif':['DejaVu Serif'],'font.size':9})

# ---------------- Fig 1: system model
fig, ax = plt.subplots(figsize=(7.2, 3.3))
L, W = 60, 12
ax.add_patch(Rectangle((0,0),L,W,fc='#f7f7f7',ec='k',lw=1))
ys=[1.5,4.5,7.5,10.5]; pax=[12,27,41,52]
ax.add_patch(FancyBboxPatch((-9.5,3.2),7.2,5.6,boxstyle='round,pad=0.2',fc='#dbe8f6',ec='#1f4e79'))
ax.text(-5.9,6.0,'BS\nbaseband\n+ ZF',ha='center',va='center',fontsize=7)
for n,(y,x) in enumerate(zip(ys,pax)):
    ax.plot([-2,L],[y,y],color='#1f4e79',lw=3,solid_capstyle='butt',zorder=2)
    ax.text(L+0.6,y,f'WG {n+1}',va='center',fontsize=7.5)
    ax.plot([-2,-2],[6,y],color='#1f4e79',lw=1)
    ax.add_patch(Rectangle((x-0.6,y-0.55),1.2,1.1,fc='#e46c0a',ec='k',zorder=4))
    ax.annotate('',xy=(x+4.5,y),xytext=(x+0.8,y),arrowprops=dict(arrowstyle='<->',color='#e46c0a',lw=1.2),zorder=5)
ax.text(pax[0]+1.2,ys[0]+0.75,r'$|\Delta x_n|\leq v_{\max}\Delta t$',fontsize=7.5,color='#a04000')
import numpy as np
rng=np.random.default_rng(3)
us=np.c_[rng.uniform(3,58,12),rng.uniform(1,11,12)]
for i,(x,y) in enumerate(us):
    ax.plot(x,y,'o',ms=6,mfc='#2e8b57',mec='k',zorder=6)
    d=rng.normal(size=2); d=d/np.linalg.norm(d)*2.2
    ax.annotate('',xy=(x+d[0],y+d[1]),xytext=(x,y),arrowprops=dict(arrowstyle='->',color='#2e8b57',lw=0.8,ls='--'))
for (x,y) in [(8,6),(20,9),(33,3),(45,8.5),(35,9.5),(55,5)]:
    ax.add_patch(Circle((x,y),0.8,fc='#8c8c8c',ec='k',zorder=3))
for (x,y) in [(16,4),(24,2.5),(38,6),(48,3),(29,10.5)]:
    ax.add_patch(Circle((x,y),0.35,fc='#c00000',ec='k',zorder=3))
# links
for n,k in [(0,1),(1,4),(2,7),(3,10)]:
    pass
ax.plot([pax[2],us[6,0]],[ys[2],us[6,1]],':',color='k',lw=0.8)
h=[plt.Line2D([],[],color='#1f4e79',lw=3,label='Dielectric waveguide (fed at $x=0$)'),
   plt.Line2D([],[],marker='s',ls='',mfc='#e46c0a',mec='k',label='Pinching antenna (finite speed)'),
   plt.Line2D([],[],marker='o',ls='',mfc='#2e8b57',mec='k',label='Mobile user + predicted motion'),
   plt.Line2D([],[],marker='o',ls='',mfc='#8c8c8c',mec='k',ms=9,label='Static blocker (shelf, 30 dB)'),
   plt.Line2D([],[],marker='o',ls='',mfc='#c00000',mec='k',label='Moving human blocker (20 dB)')]
ax.legend(handles=h,loc='upper center',bbox_to_anchor=(0.5,-0.08),ncol=3,fontsize=7,frameon=False)
ax.set_xlim(-10.5,65); ax.set_ylim(-1.8,12.8); ax.set_aspect('equal'); ax.axis('off')
ax.annotate('',xy=(60,12.4),xytext=(0,12.4),arrowprops=dict(arrowstyle='<->',lw=0.7))
ax.text(30,12.6,'$L$ = 60 m (waveguide axis, $x$)',ha='center',fontsize=7.5)
ax.text(1,-1.4,'Top view, waveguides at height $d$ = 3 m, users at $h_u$ = 1 m',fontsize=7)
plt.savefig('figures/fig1_system.png',dpi=300,bbox_inches='tight'); plt.close()

# ---------------- Fig 2: PRISM block diagram
fig, ax = plt.subplots(figsize=(7.2, 3.7))
def box(x,y,w,h,t,c):
    ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.02,rounding_size=0.08',fc=c,ec='k',lw=0.9))
    ax.text(x+w/2,y+h/2,t,ha='center',va='center',fontsize=6.9)
def arr(p,q,t=None,style='->',off=(0,0.08),rad=0.0,ls='-'):
    ax.add_patch(FancyArrowPatch(p,q,arrowstyle=style,mutation_scale=10,lw=0.9,color='k',
                 connectionstyle=f'arc3,rad={rad}',linestyle=ls))
    if t: ax.text((p[0]+q[0])/2+off[0],(p[1]+q[1])/2+off[1],t,ha='center',fontsize=6.4,style='italic')
B='#dbe8f6'; O='#fde9d9'; G='#e2efda'; Y='#fff2cc'
w=2.3
box(0.0,3.0,2.0,0.8,'Noisy user positions\n$\\tilde{\\mathbf{u}}_k(t)$',Y)
box(2.6,3.0,w,0.8,'(A) Kalman trajectory\npredictor (CV model)',B)
box(5.5,3.0,w,0.8,'Predicted positions\n$\\hat{\\mathbf{u}}_k(t+h),\\ h=1,\\dots,H$',Y)
box(2.6,1.55,w,0.8,'(B) Online radio\nblockage map (ORBM)',B)
box(5.5,1.55,w,0.8,'(C) Horizon expected-\nrate model $\\bar{r}_{n,k}(c,h)$',O)
box(8.4,2.3,2.6,1.0,'(D) Speed-constrained\ncoverage placement\n(block-coordinate ascent)',O)
box(8.4,0.1,2.6,0.8,'(E) PA actuation\n$x_n \\leftarrow x_n+\\mathrm{clip}(\\cdot,\\pm v_{\\max}\\Delta t)$',G)
box(5.5,0.1,w,0.8,'(F) PF-ZF user\nscheduling',G)
box(2.6,0.1,w,0.8,'CSI $\\mathbf{H}(t)$ at current\nPA positions',Y)
box(0.0,1.55,2.0,0.8,'PF weights\n$w_k=1/\\bar{R}_k$',Y)
arr((2.0,3.4),(2.6,3.4)); arr((4.9,3.4),(5.5,3.4))
arr((6.65,3.0),(6.65,2.35)); arr((4.9,1.95),(5.5,1.95))
arr((7.8,1.95),(8.4,2.55)); arr((9.7,2.3),(9.7,0.9),'$\\mathbf{x}^{\\star}$',off=(0.2,0))
arr((8.4,0.5),(7.8,0.5),'$\\mathbf{x}(t)$',off=(0,0.08)); arr((5.5,0.5),(4.9,0.5))
arr((3.75,0.9),(3.75,1.55),'LoS/NLoS labels',off=(0.75,-0.05))
arr((2.6,0.35),(1.0,1.55),'rate update',off=(-0.55,-0.25),rad=-0.25)
arr((2.0,2.1),(8.4,3.05),rad=-0.2,ls='--')
arr((1.0,1.55),(5.9,0.9),rad=0.15,ls='--')
ax.text(9.7,3.62,'slow loop: PA roaming\n(every $\\Delta t$ = 100 ms)',ha='center',fontsize=6.8,color='#a04000')
ax.text(6.65,-0.3,'fast loop: scheduling + blockage-map learning',ha='center',fontsize=6.8,color='#375623')
ax.set_xlim(-0.1,11.1); ax.set_ylim(-0.45,4.0); ax.axis('off')
plt.savefig('figures/fig2_prism_block.png',dpi=300,bbox_inches='tight'); plt.close()
