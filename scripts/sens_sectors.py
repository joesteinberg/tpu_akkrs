#################################################################
# imports, etc.

import sys
import numpy as np
import pandas as pd
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
from statsmodels.tsa.filters.hp_filter import hpfilter


mpl.rc('text', usetex=True)
mpl.rc('savefig',bbox='tight')
mpl.rc('savefig',format='pdf')
mpl.rc('font',**{'family':'serif','serif':['Palatino'],'size':8})
mpl.rc('font',size=8)
mpl.rc('lines',linewidth=1)

##################################################
# constants and utility functions

reform_flag=0
if len(sys.argv)>1 and sys.argv[1]=='2':
    reform_flag=2
elif len(sys.argv)>1 and sys.argv[1]=='3':
    reform_flag=3
    
inpath = '/home/joseph/Research/ongoing_projects/tpu_akkrs/model/output/'
outpath = '/home/joseph/Research/ongoing_projects/tpu_akkrs/scripts/'

alpha=0.8
colors=['#377eb8','#4daf4a','#e41a1c','#984ea3','#ff7f00']

NR = 1980
NU = 2001


def pct_chg(x):
        return (x/x.iloc[2])
    
def growth(x):
        return 100*(x/x.shift()-1.0)

def wavg(group, avg_name, weight_name):
    d = group[avg_name]
    w = group[weight_name]
    return (d * w).sum() / w.sum()


probs_baseline = np.genfromtxt(inpath + 'tpuprobs_markov_baseline.txt')
probs_sitc68 = np.genfromtxt(inpath + 'tpuprobs_markov_sitc68.txt')
probs_sitc7 = np.genfromtxt(inpath + 'tpuprobs_markov_sitc7.txt')
probs_all = [probs_baseline,probs_sitc68,probs_sitc7]
suffs = ['benchmark','SITC 6+8','SITC 7']

fig,ax = plt.subplots(1,1,figsize=(4,4))
t = range(1973,2008)

p1_out=[]
p2_out=[]

cnt=0
for probs in probs_all:
    p1 = probs.copy()
    p1[NR-1-1973:]=p1[NR-1-1973]

    ax.plot(t,p1*np.ones(len(t)),color=colors[cnt],linestyle='--',alpha=0.7,label=r'NNTR to MFN, ' + suffs[cnt])
        
    p2 = probs.copy()
    p2[0:(NR-1973)] = probs[(NR-1973)]

    ax.plot(t,p2,color=colors[cnt],alpha=0.7,label=r'MFN to NNTR, ' + suffs[cnt],linestyle='-')
                
    p1_out.append(p1*np.ones(len(t)))
    p2_out.append(p2)
    cnt = cnt+1
    
ax.axvline(NR,color='black',linestyle=':',linewidth=1,alpha=0.7)
ax.axvline(NU,color='black',linestyle=':',linewidth=1,alpha=0.7)
ax.set_xlim(1974,2007)
ax.legend(loc='upper right',prop={'size':6}) 
fig.subplots_adjust(hspace=0.2,wspace=0.25)
plt.savefig('model_fig_probabilities_sens_sectors.pdf',bbox_inches='tight')
plt.close('all')
