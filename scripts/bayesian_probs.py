#################################################################
# imports, etc.

import math
import numpy as np
import pandas as pd
from scipy.stats import beta
import matplotlib as mpl
import matplotlib.legend as mlegend
from matplotlib.patches import Rectangle
mpl.use('Agg')
import matplotlib.pyplot as plt

mpl.rc('text', usetex=True)
mpl.rc('savefig',bbox='tight')
mpl.rc('savefig',format='pdf')
mpl.rc('font',**{'family':'serif','serif':['Palatino'],'size':8})
mpl.rc('font',size=8)
mpl.rc('lines',linewidth=1)

alpha=0.7
colors=['#377eb8','#4daf4a','#e41a1c','#984ea3','#ff7f00']
markers=[None,None,'x','+','1']
linestyles=['-','--','-','-','-']

NR = 1980
NU = 2001

def tablelegend(ax, col_labels=None, row_labels=None, title_label="", *args, **kwargs):
    """
    Place a table legend on the axes.
    
    Creates a legend where the labels are not directly placed with the artists, 
    but are used as row and column headers, looking like this:
    
    title_label   | col_labels[1] | col_labels[2] | col_labels[3]
    -------------------------------------------------------------
    row_labels[1] |
    row_labels[2] |              <artists go there>
    row_labels[3] |
    
    
    Parameters
    ----------
    
    ax : `matplotlib.axes.Axes`
        The artist that contains the legend table, i.e. current axes instant.
        
    col_labels : list of str, optional
        A list of labels to be used as column headers in the legend table.
        `len(col_labels)` needs to match `ncol`.
        
    row_labels : list of str, optional
        A list of labels to be used as row headers in the legend table.
        `len(row_labels)` needs to match `len(handles) // ncol`.
        
    title_label : str, optional
        Label for the top left corner in the legend table.
        
    ncol : int
        Number of columns.
        

    Other Parameters
    ----------------
    
    Refer to `matplotlib.legend.Legend` for other parameters.
    
    """
    #################### same as `matplotlib.axes.Axes.legend` #####################
    handles, labels, extra_args, kwargs = mlegend._parse_legend_args([ax], *args, **kwargs)
    if len(extra_args):
        raise TypeError('legend only accepts two non-keyword arguments')
    
    if col_labels is None and row_labels is None:
        ax.legend_ = mlegend.Legend(ax, handles, labels, **kwargs)
        ax.legend_._remove_method = ax._remove_legend
        return ax.legend_
    #################### modifications for table legend ############################
    else:
        ncol = kwargs.pop('ncol')
        handletextpad = kwargs.pop('handletextpad', 0 if col_labels is None else -2)
        title_label = [title_label]
        
        # blank rectangle handle
        extra = [Rectangle((0, 0), 1, 1, fc="w", fill=False, edgecolor='none', linewidth=0)]
        
        # empty label
        empty = [""]
        
        # number of rows infered from number of handles and desired number of columns
        nrow = len(handles) // ncol
        
        # organise the list of handles and labels for table construction
        if col_labels is None:
            assert nrow == len(row_labels), "nrow = len(handles) // ncol = %s, but should be equal to len(row_labels) = %s." % (nrow, len(row_labels))
            leg_handles = extra * nrow
            leg_labels  = row_labels
        elif row_labels is None:
            assert ncol == len(col_labels), "ncol = %s, but should be equal to len(col_labels) = %s." % (ncol, len(col_labels))
            leg_handles = []
            leg_labels  = []
        else:
            assert nrow == len(row_labels), "nrow = len(handles) // ncol = %s, but should be equal to len(row_labels) = %s." % (nrow, len(row_labels))
            assert ncol == len(col_labels), "ncol = %s, but should be equal to len(col_labels) = %s." % (ncol, len(col_labels))
            leg_handles = extra + extra * nrow
            leg_labels  = title_label + row_labels
        for col in range(ncol):
            if col_labels is not None:
                leg_handles += extra
                leg_labels  += [col_labels[col]]
            leg_handles += handles[col*nrow:(col+1)*nrow]
            leg_labels  += empty * nrow
        
        # Create legend
        ax.legend_ = mlegend.Legend(ax, leg_handles, leg_labels, ncol=ncol+int(row_labels is not None), handletextpad=handletextpad, handlelength=2.25, columnspacing=0.8, prop={'size':6},**kwargs)
        ax.legend_._remove_method = ax._remove_legend
        return ax.legend_

# bootstrap confidence intervals
def bootstrap(data, n=50, func=np.mean):
    """
    Generate `n` bootstrap samples, evaluating `func`
    at each resampling. `bootstrap` returns a function,
    which can be called to obtain confidence intervals
    of interest.
    """
    simulations = list()
    sample_size = len(data)
    xbar_init = np.mean(data)
    for c in range(n):
        itersample = np.random.choice(data, size=sample_size, replace=True)
        simulations.append(func(itersample))
    simulations.sort()
    def ci(p):
        """
        Return 2-sided symmetric confidence interval specified
        by p.
        """
        u_pval = (1+p)/2.
        l_pval = (1-u_pval)
        l_indx = int(np.floor(n*l_pval))
        u_indx = int(np.floor(n*u_pval))
        return(simulations[l_indx],simulations[u_indx])
    return(ci)

# prior in 1970
priors0 = [(1,1),(2,2),(8,8),(8,4),(0.1,0.1)]
mu0 = [[] for p in priors0]
for i in range(len(priors0)):
    a = priors0[i][0]
    b = priors0[i][1]
    
    # 1970 prior + posteriors during 1971-1979
    for t in range(0,9):
        mu0[i].append(float(beta.stats(t+a,b,moments='m')))

mu1_model = [0.715190431,
             0.802387348,
             0.738795119,
             0.728883531,
             0.705237034,
             0.666131652,
             0.59665106,
             0.472445679,
             0.396320301,
             0.26667528,
             0.183683168,
             0.127258675,
             0.08902666,
             0.079660018,
             0.106954907,
             0.13506244,
             0.148278937,
             0.138940624,
             0.105917082,
             0.068812713,
             0.052700157,
             0.053781898,
             0.062664147,
             0.075994064,
             0.07326541,
             0.061294273,
             0.039430122,
             0.03397739]

x = mu1_model[0]
# a/(a+b)=x --> a = a*x + b*x --> a(1-x) = b*x --> a = b * x/(1-x)
priors1 = [(b*x/(1-x),b) for b in np.linspace(1,5,5)]
mu1 = [[] for p in priors1]
for i in range(len(priors1)):

    a = priors1[i][0]
    b = priors1[i][1]

    # posteriors during 1980-2008
    for t in range(0,28):   
        mu1[i].append(float(beta.stats(a,t+b,moments='m')))

theta = np.linspace(0,1,25)
fig1,axes1=plt.subplots(1,1,figsize=(4,4),sharex=False,sharey=False)
fig2,axes2=plt.subplots(1,1,figsize=(4,4),sharex=False,sharey=False)

lns2=[]
lns1=[]

lns2 = lns2 + axes2.plot(range(1980,2008),mu1_model,color='black',linewidth=2,marker='o',markersize=5,label='Model')
for i in range(len(priors1)):
    a = priors1[i][0]
    b = priors1[i][1]
    lns1 = lns1 + axes1.plot(theta,beta.pdf(theta,a,b),linestyle=linestyles[i],label='b=%0.1f'%(b),alpha=alpha,linewidth=1,color=colors[i],marker=markers[i])
    lns2 = lns2 + axes2.plot(range(1980,2008),mu1[i],linestyle=linestyles[i],label='b=%0.1f'%(b),alpha=alpha,linewidth=1,color=colors[i],marker=markers[i])
    
axes1.axvline(x,color='black',linestyle=':',linewidth=1,alpha=0.7,zorder=1)
axes2.axvline(NU,color='black',linestyle=':',linewidth=1,alpha=0.7,zorder=1)
axes2.set_xlim(1980,2008)

axes1.legend(loc='best',prop={'size':6})
axes2.legend(loc='best',prop={'size':6})
fig1.subplots_adjust(hspace=0.2,wspace=0.25)
fig2.subplots_adjust(hspace=0.2,wspace=0.25)

plt.sca(axes1)
plt.savefig('bayesian_probs_priors.pdf',bbox_inches='tight')
df1 = pd.DataFrame(dict(zip(['omega12'] + ['p(omega12|'+l.get_label()+')' for l in lns1],[theta] + [l.get_ydata() for l in lns1])))
df1.to_csv('model_fig_bayes_priors.csv',index=False)

plt.sca(axes2)
plt.savefig('bayesian_probs_posteriors.pdf',bbox_inches='tight')
plt.close('all')

df2 = pd.DataFrame(dict(zip(['Year'] + [l.get_label() for l in lns2],[range(1980,2008)] + [l.get_ydata() for l in lns2])))
df2.to_csv('model_fig_bayes_posteriors.csv',index=False)


