#!/usr/bin/env python3
"""Publication-size spectral figure from saved CSVs. No solver or data fitting."""
from pathlib import Path
import csv, hashlib, json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import NullLocator

HERE = Path(__file__).resolve().parent
DATA_FILES = ['full_spectra.csv', 'spectral_metrics.csv']
ARCHIVED_DATA_FILES = ['weak_directions.csv']
def read(name):
    with (HERE/name).open() as stream:
        return list(csv.DictReader(stream))
spec, metrics = [read(name) for name in DATA_FILES]
plt.rcParams.update({
    'font.family':'sans-serif', 'font.sans-serif':['Liberation Sans'],
    'font.size':8.5, 'axes.labelsize':9, 'axes.titlesize':9,
    'xtick.labelsize':8, 'ytick.labelsize':8, 'legend.fontsize':8,
    'axes.linewidth':.7, 'mathtext.fontset':'custom',
    'mathtext.rm':'Liberation Sans', 'mathtext.it':'Liberation Sans:italic',
    'mathtext.bf':'Liberation Sans:bold', 'mathtext.sf':'Liberation Sans',
    'mathtext.cal':'STIXNonUnicode:italic', 'mathtext.fallback':'stix',
    'pdf.fonttype':42, 'ps.fonttype':42, 'path.simplify':False,
    'savefig.facecolor':'white'})
betas = [0,.5,1]
colors = {0:'#0072B2', .5:'#D55E00', 1:'#666666'}
styles = {0:'--', .5:'-', 1:':'}
markers = {0:'o', .5:'s', 1:'^'}
cases = ['ovt_varnu_f1','ovt_varnu_f1p5','ovt_nu25_f1','ovt_nu49_f1']
case_labels = [r'Variable $\nu$'+'\n1 Hz', r'Variable $\nu$'+'\n1.5 Hz',
               r'$\nu=0.25$'+'\n1 Hz', r'$\nu=0.49$'+'\n1 Hz']
# Explicit gutters retain 8--9 pt text at the actual manuscript width.
fig = plt.figure(figsize=(119/25.4,58/25.4))
ax_a = fig.add_axes([.11,.27,.36,.56])
ax_b = fig.add_axes([.61,.27,.36,.56])
# (a) The same lowest 40 eigenvalues, with method styles shared with (b).
for case in ['ovt_nu25_f1','ovt_nu49_f1']:
    for beta in betas:
        rows = [d for d in spec if d['case']==case and float(d['beta'])==beta and int(d['index'])<=40]
        assert len(rows)==40
        ax_a.semilogy([int(d['index']) for d in rows],
                      [float(d['eigenvalue']) for d in rows],
                      color=colors[beta],ls=styles[beta],lw=1.1,
                      marker=markers[beta],markevery=[0,8,16,24,32,39],
                      ms=3,mfc='white',mew=.65)
ax_a.set(xlabel='Eigenvalue index',ylabel=r'$\lambda_i(\widehat{\mathbf{H}}_W)$',
         xlim=(0,41),ylim=(1e-7,2e-3))
ax_a.set_xticks([1,10,20,30,40])
ax_a.set_yticks([1e-7,1e-6,1e-5,1e-4,1e-3])
ax_a.yaxis.set_minor_locator(NullLocator())
ax_a.text(.06,.98,r'$\nu=0.25$',transform=ax_a.transAxes,va='top')
ax_a.text(.06,.46,r'$\nu=0.49$',transform=ax_a.transAxes,va='top')
# (b) Categorical columns and positive-spectrum condition numbers. Horizontal
# dodge separates nearby beta=0/1 values without connecting unrelated cases.
x = np.arange(4)
offsets = {0:-.16,.5:0,1:.16}
plotted_condition_numbers = []
for beta in betas:
    values = [float(next(d['kappa'] for d in metrics if d['case']==case and float(d['beta'])==beta)) for case in cases]
    ax_b.plot(x+offsets[beta],values,ls='none',marker=markers[beta],
              ms=4.6,color=colors[beta],mfc='white',mew=1)
    plotted_condition_numbers.extend({'case':case,'beta':beta,'kappa':value}
                                     for case,value in zip(cases,values))
ax_b.set_yscale('log')
ax_b.set(ylabel=r'Condition number $\kappa_+$',
         xlim=(-.5,3.5),ylim=(1e3,1e7))
ax_b.set_yticks([1e3,1e4,1e5,1e6,1e7])
ax_b.yaxis.set_minor_locator(NullLocator())
ax_b.set_xticks(x,case_labels)
for boundary in [.5,1.5,2.5]:
    ax_b.axvline(boundary,color='.88',lw=.45,zorder=0)
for label,ax in zip(['(a)','(b)'],[ax_a,ax_b]):
    ax.text(0,1.045,label,transform=ax.transAxes,ha='left',va='bottom',fontsize=9)
    ax.tick_params(direction='in',width=.65,length=3,top=True,right=True)
ax_b.tick_params(axis='x',length=0,pad=6)
# Tiny horizontal offsets keep the original two-line labels clear at 8 pt.
from matplotlib.transforms import ScaledTranslation
for tick,offset in zip(ax_b.get_xticklabels(),[-8,1,6,9]):
    tick.set_transform(tick.get_transform()+ScaledTranslation(offset/72,0,fig.dpi_scale_trans))
handles = [Line2D([],[],color=colors[beta],ls=styles[beta],lw=1.2,
                  marker=markers[beta],ms=4,mfc='white',mew=.8,label=label)
           for beta,label in [(0,r'Weighted ($\beta=0$)'),(.5,r'Weighted ($\beta=0.5$)'),(1,r'Weighted ($\beta=1$)')]]
fig.legend(handles=handles,loc='upper center',bbox_to_anchor=(.52,1.005),ncol=3,
           frameon=False,handlelength=2.0,columnspacing=1.0,handletextpad=.6,fontsize=8.5)
for suffix in ['pdf','svg','png']:
    fig.savefig(HERE/f'Fig_complete_defect_spectrum.{suffix}',dpi=220)
plt.close(fig)
(HERE/'FIGURE_MANIFEST.json').write_text(json.dumps({
    'data_files':{name:hashlib.sha256((HERE/name).read_bytes()).hexdigest()
                  for name in DATA_FILES+ARCHIVED_DATA_FILES},
    'plotted_data_files':DATA_FILES,
    'archived_unplotted_data_files':ARCHIVED_DATA_FILES,
    'panels':['lowest 40 eigenvalues for controlled nu pairs',
              'categorical cases on x; log-scale positive-spectrum condition numbers on y'],
    'data_selection':'The same six lowest-40 eigenvalue curves and all twelve condition numbers for the four predeclared cases. No fitting, smoothing or new solver work.',
    'condition_points':plotted_condition_numbers,
    'condition_metric_label':'Condition number kappa_+',
    'condition_plot':{'x_axis':'four categorical cases','y_axis':'positive-spectrum condition number',
                      'y_scale':'log','horizontal_offsets':offsets,
                      'marker_fill':'white','connect_cases':False},
    'layout':'CMAME-style landscape side-by-side panels, shared top legend and two-line category labels; 119mm width and minimum 8pt ordinary lettering',
    'size_inches':[119/25.4,58/25.4],
    'pdf':'Fig_complete_defect_spectrum.pdf'},indent=2)+'\n')
