"""Assemble the report from the saved results and published numerical tables."""

from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path
import shutil

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

ROOT = Path('D:/SummerResearch')
OUT = ROOT / 'output/pdf/synthetic_benchmark'
RUN = ROOT / 'output/simulated_all_20260930'
DATASETS = ['gaussian', 'grid', 'ring', 'asia', 'alarm', 'child', 'insurance']
LABELERS = {'gaussian':'GNB', 'categorical':'CNB', 'pca_gmm':'PCA--GMM', 'rf':'RF', 'xgb':'XGB', 'dnn':'DNN'}
COLORS = {'original':'#7e8da3', 'baseline':'#263445', 'joint':'#008696', 'features':'#a565cc'}
METRICS = ['l_syn','l_test','accuracy','macro_f1','h_star_agreement']

def num(value, places=3, percent=False):
    d = Decimal(str(value)) * (100 if percent else 1)
    return str(d.quantize(Decimal(10) ** -places, rounding=ROUND_HALF_UP))

def method_name(method, plot=False):
    if method == 'original':
        return 'Original'
    if '-' not in method:
        return method.upper()
    generator, source, labeler = method.split('-')
    label = LABELERS[labeler].replace('--','-') if plot else LABELERS[labeler]
    return f'{generator.upper()} / {"J" if source=="joint" else "F"} / {label}'

def group(method):
    return 'original' if method == 'original' else method.split('-')[1] if '-' in method else 'baseline'

def table(headers, rows, spec, caption, label, size='small'):
    body = '\n'.join(' & '.join(row) + r'\\' for row in rows)
    return ('\\begin{table}[H]\\centering\n\\caption{' + caption + '}\\label{' + label + '}\n'
            '\\' + size + '\n\\begin{tabular}{' + spec + '}\\toprule\n' +
            ' & '.join(headers) + r'\\\midrule' + '\n' + body +
            '\n\\bottomrule\\end{tabular}\n\\end{table}')

MAIN = [
['Identity','-2.61','-2.61','-9.33','-9.36'],
['CLBN','-3.06','-7.31','-10.66','-9.92'],
['PrivBN','-3.38','-12.42','-12.97','-10.90'],
['MedGAN','-7.27','-60.03','-11.14','-12.15'],
['VEEGAN','-10.06','-4.22','-15.40','-13.86'],
['TableGAN','-8.24','-4.12','-11.84','-10.47'],
['TVAE','-2.65','-5.42','-6.76','-9.59'],
['CTGAN','-5.72','-3.40','-11.67','-10.60']]
PAPER_GM = {
'Identity':['-3.06','-3.06','-3.06','-3.07','-1.70','-1.70'],
'CLBN':['-3.68','-8.62','-3.76','-11.60','-1.75','-1.70'],
'PrivBN':['-4.33','-21.67','-3.98','-13.88','-1.82','-1.71'],
'MedGAN':['-10.04','-62.93','-9.45','-72.00','-2.32','-45.16'],
'VEEGAN':['-9.81','-4.79','-12.51','-4.94','-7.85','-2.92'],
'TableGAN':['-8.70','-4.99','-9.64','-4.70','-6.38','-2.66'],
'TVAE':['-2.86','-11.26','-3.41','-3.20','-1.68','-1.79'],
'CTGAN':['-5.63','-3.69','-8.11','-4.31','-3.43','-2.19']}
PAPER_BN = {
'Identity':['-2.23','-2.24','-10.3','-10.3','-12.0','-12.0','-12.8','-12.9'],
'CLBN':['-2.44','-2.27','-12.4','-11.2','-12.6','-12.3','-15.2','-13.9'],
'PrivBN':['-2.28','-2.24','-11.9','-10.9','-12.3','-12.2','-14.7','-13.6'],
'MedGAN':['-2.81','-2.59','-10.9','-14.2','-14.2','-15.4','-16.4','-16.4'],
'VEEGAN':['-8.11','-4.63','-17.7','-14.9','-17.6','-17.8','-18.2','-18.1'],
'TableGAN':['-3.64','-2.77','-12.7','-11.5','-15.0','-13.3','-16.0','-14.3'],
'TVAE':['-2.31','-2.27','-11.2','-10.7','-12.3','-12.3','-14.7','-14.2'],
'CTGAN':['-2.56','-2.31','-14.2','-12.6','-13.4','-12.7','-16.5','-14.8']}

r = pd.read_csv(RUN/'per_run.csv')
s = pd.read_csv(RUN/'summary.csv').set_index(['dataset','method'])
assert len(r)==378 and len(s)==189 and np.isfinite(r[METRICS]).all().all()
expected = r.groupby(['dataset','method'], sort=False)[METRICS+['support_violation_rate','accuracy_ceiling']].mean()
pd.testing.assert_frame_equal(s, expected, check_exact=False, rtol=1e-12, atol=1e-12)
arch = pd.read_csv(ROOT/'audit/remote_simulated_methods_per_run.csv')
arch = arch[(arch.benchmark=='paper') & arch.method.isin(['identity','ctgan','tvae'])]
assert len(arch)==42
arch_mean = arch.groupby(['dataset','method'])[['l_syn','l_test']].mean()

sections = {}
sections['PAPER_AVERAGES'] = table(['Method',r'GM $\Lsyn$',r'GM $\Ltest$',r'BN $\Lsyn$',r'BN $\Ltest$'],
MAIN,'lrrrr',r'Main paper Table 2: simulated-data averages, transcribed as published~\cite{xu2019}.','tab:paperaverages')

rows=[]
for j,d in enumerate(DATASETS[3:]):
    for m in ['original','ctgan','tvae']:
        paper=PAPER_BN['Identity' if m=='original' else m.upper()]
        ours=s.loc[(d,m)]
        rows.append([d.title(),method_name(m),paper[2*j],num(ours.l_syn),paper[2*j+1],num(ours.l_test)])
sections['BN_COMPARISON']=table(['Dataset','Method',r'Paper $\Lsyn$',r'Ours $\Lsyn$',r'Paper $\Ltest$',r'Ours $\Ltest$'],
rows,'llrrrr',r'Current BN baseline means versus supplement Table 3~\cite{xusupplement}. Both BN scores use $\log(p+10^{-8})$; our refit uses $\alpha=1/2$.','tab:bncomparison')

rows=[]
for j,d in enumerate(['grid','gridr','ring']):
    for m in ['identity','ctgan','tvae']:
        paper=PAPER_GM['Identity' if m=='identity' else m.upper()]
        ours=arch_mean.loc[(d,m)]
        rows.append([{'grid':'Grid','gridr':'GridR','ring':'Ring'}[d],m.title() if m=='identity' else m.upper(),
                     paper[2*j],num(ours.l_syn),paper[2*j+1],num(ours.l_test)])
sections['ARCHIVED_GM']=table(['Dataset','Method',r'Paper $\Lsyn$',r'Archive $\Lsyn$',r'Paper $\Ltest$',r'Archive $\Ltest$'],
rows,'llrrrr',r'Archived reproduction of the original two-column GM benchmark. The archive is separate from the current mixed-data run.','tab:archivedgm')

rows=[]
for d in DATASETS:
    for m in ['original','ctgan','tvae']:
        v=s.loc[(d,m)]
        rows.append([d.title(),method_name(m),*[num(v[k],2,True) for k in ['accuracy','macro_f1','h_star_agreement','support_violation_rate']]])
sections['BASELINE_UTILITY']=table(['Dataset','Method',r'Accuracy (\%)',r'Macro F1 (\%)',r'$A_{h^*}$ (\%)',r'$v$ (\%)'],
rows,'llrrrr','Current generator baselines and original-data training: two-seed mean prediction and exact support metrics.','tab:baselines')

rows=[]
for d in DATASETS:
    v=s.loc[d].drop(index='original')
    m=v.accuracy.idxmax()
    row=v.loc[m]
    rows.append([d.title(),method_name(m),num(s.loc[(d,'original'),'accuracy'],2,True),num(row.accuracy,2,True),
                 num(row.h_star_agreement,2,True),num(row.l_test),num(row.support_violation_rate,2,True)])
sections['ACCURACY_SELECTION']=table(['Dataset','Selected method',r'Original acc.',r'Synthetic acc.',r'$A_{h^*}$',r'$\Ltest$',r'$v$'],
rows,'llrrrrr','Highest observed mean synthetic accuracy per dataset. Prediction and support columns are percentages.','tab:accuracyselection','footnotesize')

gmheaders=['Method']+[fr'{d} $L_{{{m}}}$' for d in ['Grid','GridR','Ring'] for m in ['s','t']]
bnheaders=['Method']+[fr'{d} $L_{{{m}}}$' for d in ['Asia','Alarm','Child','Insurance'] for m in ['s','t']]
sections['PAPER_FULL']=table(gmheaders,[[k]+v for k,v in PAPER_GM.items()],'lrrrrrr',
r'Original GM supplement values. $L_s=\Lsyn$ and $L_t=\Ltest$. The last printed GM row is identified as CTGAN as explained in the main text.','tab:papergmfull')+'\n'+table(
bnheaders,[[k]+v for k,v in PAPER_BN.items()],'lrrrrrrrr',r'Original BN supplement values. The published TGAN row is labeled CTGAN here.','tab:paperbnfull','footnotesize')

full=[]
for i,d in enumerate(DATASETS):
    if i:
        full.append(r'\clearpage')
    full.append(r'\subsection{'+d.title()+'}')
    rows=[]
    for m in r.loc[r.dataset==d,'method'].unique():
        v=s.loc[(d,m)]
        rows.append([method_name(m),num(v.l_syn),num(v.l_test),*[num(v[k],2,True) for k in ['accuracy','macro_f1','h_star_agreement','support_violation_rate']]])
    full.append(table(['Method',r'$\Lsyn$',r'$\Ltest$',r'Acc. (\%)',r'F1 (\%)',r'$A_{h^*}$ (\%)',r'$v$ (\%)'],
                      rows,'lrrrrrr',d.title()+r': all 27 method means. F1 denotes macro F1.','tab:all-'+d,'small'))
    ceiling=s.loc[(d,'original'),'accuracy_ceiling']
    full.append('Expected Bayes accuracy reference: '+num(ceiling,2,True)+r'\%. The reference uses exact test-feature posteriors for BN data and $1-\eta$ for mixed data.')
sections['FULL_RESULTS']='\n'.join(full)

plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'pdf.fonttype':42})
figure_sections=[]
for d in DATASETS:
    data=r[r.dataset==d]
    methods=data.method.unique()
    fig,axs=plt.subplots(1,5,figsize=(11.5,7.3))
    fig.subplots_adjust(left=.205,right=.985,bottom=.16,top=.83,wspace=.38)
    fig.suptitle(d.title()+': likelihood and predictive utility',x=.02,y=.985,ha='left',fontsize=14,weight='bold')
    titles=[r'$L_{\rm syn}$',r'$L_{\rm test}$','Accuracy','Macro F1',r'Agreement with $h^*$']
    for j,(ax,metric,title) in enumerate(zip(axs,METRICS,titles)):
        scale=1 if j<2 else 100
        for gen,style in [('ctgan','--'),('tvae',':')]:
            ax.axvline(data.loc[data.method==gen,metric].mean()*scale,color=COLORS['baseline'],ls=style,lw=.9,alpha=.7)
        for k,m in enumerate(methods):
            vals=data.loc[data.method==m,metric].to_numpy()*scale
            color=COLORS[group(m)]
            ax.plot([vals.min(),vals.max()],[k,k],color=color,lw=.65,alpha=.65)
            ax.scatter(vals,[k]*2,s=24,facecolors='white',edgecolors=color,lw=.9,zorder=3)
            mean=vals.mean()
            ax.scatter(mean,k,s=10,color=color,zorder=4)
            ax.annotate(f'{mean:.2f}' if j<2 else f'{mean:.1f}',(mean,k),xytext=(3,4),textcoords='offset points',fontsize=7)
            if k in [1,2,8,14,15,21]:
                ax.axhline(k-.5,color='#e1e6ed',lw=.6)
        ax.set_ylim(len(methods)-.4,-.6)
        ax.set_yticks(np.arange(len(methods)),[method_name(m,True) for m in methods] if j==0 else [])
        ax.tick_params(axis='both',labelsize=8,length=2)
        ax.grid(axis='x',color='#e1e6ed',lw=.5)
        for edge in ['top','right','left']:
            ax.spines[edge].set_visible(False)
        ax.set_title(title,fontsize=10,pad=8)
        ax.set_xlabel('Nats / row' if j<2 else 'Percent',fontsize=8)
        if j<2:
            ax.margins(x=.36)
            ax.locator_params(axis='x',nbins=4)
        else:
            ax.set_xlim(0,110)
            ax.set_xticks([0,50,100])
    axs[2].axvline(data.accuracy_ceiling.mean()*100,color='#c47820',ls='-.',lw=1)
    handles=[Line2D([],[],color=c,marker='o',ls='',markersize=4,label=label) for c,label in
             [(COLORS['original'],'Original'),(COLORS['baseline'],'Generated target'),(COLORS['joint'],'Joint + labels'),(COLORS['features'],'Features + labels')]]
    fig.legend(handles=handles,loc='upper left',bbox_to_anchor=(.02,.946),ncol=4,frameon=False,fontsize=9)
    handles=[Line2D([],[],color=COLORS['baseline'],ls=ls,label=label) for ls,label in [('--','CTGAN baseline'),(':','TVAE baseline')]]
    handles.append(Line2D([],[],color='#c47820',ls='-.',label='Expected Bayes accuracy'))
    fig.legend(handles=handles,loc='upper left',bbox_to_anchor=(.02,.905),ncol=3,frameon=False,fontsize=9)
    fig.text(.02,.075,'J: joint relabeling; F: feature-only synthesis and labeling. Open markers: seeds 42 and 43; filled markers: means.',fontsize=9)
    note='BN likelihoods use log(p + 1e-8); exact impossible-row fractions are reported in the tables.' if d in DATASETS[3:] else 'Mixed likelihoods are exact log densities. The expected Bayes accuracy is 90%.'
    fig.text(.02,.04,note,fontsize=9)
    fig.savefig(OUT/'figures'/f'{d}.pdf',bbox_inches='tight')
    fig.savefig(ROOT/'tmp/pdfs/synthetic_benchmark'/f'{d}_figure.png',dpi=140,bbox_inches='tight')
    plt.close(fig)
    figure_sections.append(r'\begin{landscape}\pdfpageattr{/Rotate 90}'+'\n'+r'\begin{figure}[H]\centering'+'\n'+
                           r'\includegraphics[width=\linewidth]{figures/'+d+r'.pdf}'+'\n'+
                           r'\caption{'+d.title()+r': all methods and both seeds. Likelihood panels and prediction panels use the definitions in Section 4.}'+'\n'+
                           r'\label{fig:'+d+r'}\end{figure}'+'\n'+r'\end{landscape}\pdfpageattr{}')
sections['FIGURES']='\n'.join(figure_sections)

template=(ROOT/'tmp/pdfs/synthetic_benchmark/report_template.tex').read_text(encoding='utf-8')
for key,value in sections.items():
    token='@@'+key+'@@'
    assert template.count(token)==1
    template=template.replace(token,value)
assert '@@' not in template
(OUT/'synthetic_benchmark_report.tex').write_text(template,encoding='utf-8')
for name in ['per_run.csv','summary.csv','config.json']:
    shutil.copyfile(RUN/name,OUT/'data'/name)
arch.to_csv(OUT/'data/archived_paper_baselines.csv',index=False)
print('Wrote source, all 189 method means, published comparison tables, and seven vector figures.')
