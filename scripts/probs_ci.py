#################################################################
# imports, etc.

import numpy as np
import pandas as pd
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

mpl.rc('savefig',bbox='tight')
mpl.rc('savefig',format='pdf')

##################################################
# constants and utility functions
    
inpath = '/home/joseph/Research/ongoing_projects/tpu_akkrs/model/output/'
outpath = '/home/joseph/Research/ongoing_projects/tpu_akkrs/scripts/'

alpha=0.8
colors=['#377eb8','#4daf4a','#e41a1c','#984ea3','#ff7f00']

NR = 1980
NU = 2001


probs_baseline = np.genfromtxt(inpath + 'tpuprobs_markov_baseline.txt')
probs_lo = np.genfromtxt(inpath + 'tpuprobs_markov_ci_lower.txt')
probs_hi = np.genfromtxt(inpath + 'tpuprobs_markov_ci_upper.txt')
probs_all = [probs_baseline,probs_lo,probs_hi]
suffs = ['benchmark','upper bound','lower bound']

fig, ax = plt.subplots(figsize=(16, 10))
ax.tick_params(axis='both', labelsize=18)
ax.yaxis.label.set_size(18)
sns.despine()
lw=3
tw=20

t=range(1974,2009)

cnt=0
for probs in probs_all:
    p1 = probs.copy()
    p1[NR-1-1973:]=p1[NR-1-1973]

    p2 = probs.copy()
    p2[0:(NR-1973)] = probs[(NR-1973)]

    if(cnt==0):
        ax.plot(t,p1*np.ones(len(t)),color=colors[0],linestyle='--',alpha=0.7,label=r'NNTR to MFN',lw=lw)
        ax.plot(t,p2,color=colors[2],alpha=0.7,label=r'MFN to NNTR',lw=lw)
    else:
        ax.plot(t,p1*np.ones(len(t)),color=colors[0],linestyle='--',linewidth=1,alpha=0.3)
        ax.plot(t,p2,color=colors[2],alpha=0.3,linewidth=1)
        
    cnt = cnt+1


ax.axvline(NR,color='black',linestyle='--',linewidth=1,alpha=0.7)
ax.axvline(NU,color='black',linestyle='--',linewidth=1,alpha=0.7)

ax.set_xlim(1974,2007)
ax.legend(loc='upper right',prop={'size':tw}) 
plt.savefig('model_fig_probabilities_ci.pdf',bbox_inches='tight')
plt.close('all')
