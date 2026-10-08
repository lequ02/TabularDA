"""Prepare report figures from the statistics calculated on the server."""
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
TABLES = Path(__file__).parent / 'results'
OUT = ROOT / 'output/statistical_review_20261006'
OUT.mkdir(exist_ok=True, parents=True)

def read(name):
    return list(csv.DictReader((TABLES/name).open()))

plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,
                     'axes.spines.right':False,'axes.titleweight':'bold','savefig.dpi':220})
tests = read('contrast_tests.csv')
fig, axes = plt.subplots(1,2,figsize=(11.8,3.8))
groups = [('all_six','All six labelers'),('rf_xgb_dnn','RF / XGB / DNN'),('dnn','DNN')]
for ax,domain,title,limit in zip(axes,['real_core','simulated'],['Real data: macro F1, 4 datasets','Simulation: accuracy, 7 tasks'],[2.5,.6]):
    for i,(group,label) in enumerate(groups):
        r=next(r for r in tests if r['analysis']==domain and r['labeler_group']==group and r['contrast']=='full_minus_xonly')
        mean,lo,hi=map(float,[r['mean'],r['ci_low'],r['ci_high']])
        ax.errorbar(mean,2-i,xerr=np.array([[mean-lo],[hi-mean]]),fmt='o',color='#176d81',capsize=4,markersize=7)
        ax.annotate(f'{mean:+.2f}',(mean,2-i),xytext=(0,12),textcoords='offset points',ha='center',fontsize=9)
    ax.axvline(0,color='#73838c',ls='--',lw=1)
    ax.set(yticks=[2,1,0],yticklabels=[g[1] for g in groups],xlim=(-limit,limit),ylim=(-.5,2.65))
    ax.set_title(title,fontsize=12,pad=15)
    ax.set_xlabel('Full minus X-only (percentage points)')
    ax.grid(axis='x',alpha=.17)
fig.suptitle('Including Y in generator training has no clear utility advantage',fontsize=14,y=1.01)
fig.text(.5,-.035,'Dots are equal-dataset means. Bars are unadjusted 95% paired-t intervals across datasets.',ha='center',fontsize=9,color='#52616b')
fig.tight_layout()
for ext in ['png','svg']:fig.savefig(OUT/f'full_vs_xonly.{ext}',bbox_inches='tight')
plt.close(fig)

effects = read('dataset_effects.csv')
labels=['gaussian','categorical','pca_gmm','rf','xgb','dnn']
names=['Gaussian NB','Categorical NB','PCA/GMM','RF','XGB','DNN']
orders=[['adult','census_kdd','covertype','mnist28'],['gaussian','grid','ring','asia','alarm','child','insurance']]
fig,axes=plt.subplots(1,2,figsize=(12.5,5.0),gridspec_kw={'width_ratios':[1,1]})
for ax,domain,order,title in zip(axes,['real_core','simulated'],orders,['Real data: macro F1','Simulation: accuracy']):
    matrix=np.array([[float(next(r['difference'] for r in effects if r['analysis']==domain and r['dataset']==dataset
                                 and r['labeler_group']==label and r['contrast']=='full_hybrid_minus_generated'))
                      for label in labels] for dataset in order])
    im=ax.imshow(matrix,cmap='RdBu',vmin=-35,vmax=35,aspect='auto')
    for (y,x),value in np.ndenumerate(matrix):
        ax.text(x,y,f'{value:+.1f}',ha='center',va='center',fontsize=9,color='white' if abs(value)>20 else '#142b3b')
    display={'census_kdd':'Census KDD*','mnist28':'MNIST28','covertype':'Covertype','adult':'Adult'}
    ax.set(xticks=range(6),xticklabels=names,yticks=range(len(order)),yticklabels=[display.get(x,x.title()) for x in order])
    plt.setp(ax.get_xticklabels(),rotation=35,ha='right',rotation_mode='anchor')
    ax.set_title(title,fontsize=12,pad=14)
    ax.tick_params(length=0)
fig.suptitle('Hybrid gains depend strongly on the labeler',fontsize=15,y=1.02)
fig.subplots_adjust(wspace=.3,bottom=.23,right=.9)
cax=fig.add_axes([.93,.24,.016,.59]);fig.colorbar(im,cax=cax,label='Hybrid minus generated-target baseline (pp)')
fig.text(.5,.02,'Full-table relabeling; seeds, generators and modes averaged within each dataset. *Census: matched CTGAN only.',ha='center',fontsize=9,color='#52616b')
for ext in ['png','svg']:fig.savefig(OUT/f'labeler_effects.{ext}',bbox_inches='tight')
plt.close(fig)

pairs=read('paired_differences.csv');values=defaultdict(list)
for r in pairs:
    if r['labeler_group']=='dnn' and r['contrast']=='full_hybrid_minus_generated':
        if (r['analysis']=='simulated' and r['metric']=='accuracy') or (r['analysis']=='simulated_secondary' and r['metric']=='l_test'):
            values[(r['dataset'],r['generator'],r['metric'])].append(float(r['difference']))
fig,ax=plt.subplots(figsize=(9.4,4.5))
abbreviations={'gaussian':'Gaussian','grid':'Grid','ring':'Ring','asia':'Asia','alarm':'Alarm','child':'Child','insurance':'Insurance'}
offsets={('ring','tvae'):(-5,-17),('ring','ctgan'):(-45,12),('grid','tvae'):(-5,8),
         ('child','ctgan'):(7,8),('insurance','ctgan'):(7,-15),('child','tvae'):(12,22),('insurance','tvae'):(15,-20)}
for generator,color,marker in [('ctgan','#b46c2b','o'),('tvae','#176d81','s')]:
    for dataset in orders[1]:
        x=sum(values[(dataset,generator,'l_test')])/2
        y=sum(values[(dataset,generator,'accuracy')])/2
        ax.scatter(x,y,color=color,marker=marker,s=55,label=generator.upper() if dataset=='gaussian' else None,zorder=3)
        ax.annotate(abbreviations[dataset],(x,y),xytext=offsets.get((dataset,generator),(6,6)),textcoords='offset points',fontsize=9,color=color)
ax.axvline(0,color='#73838c',ls='--',lw=1);ax.axhline(0,color='#73838c',ls='--',lw=1)
ax.set(xlabel='Change in L_test density-refit score (nats per row)',ylabel='Change in downstream accuracy (pp)',xlim=(-.95,.39),ylim=(-1.5,29))
ax.set_title('Higher predictive utility can accompany worse density fidelity',fontsize=13,pad=14)
ax.grid(alpha=.15);ax.legend(loc='upper left',frameon=False)
fig.text(.5,-.015,'Full-table DNN relabeling, mean of seeds 42/43. BN scores use log(p + 1e-8); mixed densities are exact.',ha='center',fontsize=9,color='#52616b')
fig.tight_layout()
for ext in ['png','svg']:fig.savefig(OUT/f'utility_and_density.{ext}',bbox_inches='tight')
plt.close(fig)
print(OUT)
