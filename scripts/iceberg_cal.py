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
import patsy
from numpy.linalg import pinv
import seaborn as sns

#mpl.rc('text', usetex=True)
mpl.rc('savefig',bbox='tight')
mpl.rc('savefig',format='pdf')
#mpl.rc('font',**{'family':'serif','serif':['Palatino'],'size':8})
#mpl.rc('font',size=10)
#mpl.rc('lines',linewidth=1)

##################################################
# constants and utility functions

reform_flag=3
    
inpath = '/home/joseph/Research/ongoing_projects/tpu_akkrs/model/output/'
outpath = '/home/joseph/Research/ongoing_projects/tpu_akkrs/scripts/'

alpha=0.8
colors=['#377eb8','#e41a1c','#4daf4a','#984ea3']

NR = 1980
NU = 2001

agg_fns={'f':[('nf',lambda x:x.nunique())],
         'v':[('exports',lambda x:x.sum())]}

def reset_multiindex(df,n,suff):
    levels=df.columns.levels
    labels=df.columns.labels
    df.columns=levels[0][labels[0][0:n]].tolist()+[s+suff for s in levels[1][labels[1][n:]].tolist()]
    return df

def pct_chg(x):
        return (x/x.iloc[2])
    
def growth(x):
        return 100*(x/x.shift()-1.0)

def wavg(group, avg_name, weight_name):
    d = group[avg_name]
    w = group[weight_name]
    return (d * w).sum() / w.sum()

##################################################
# load data
print('\tloading data')

suff=''
suff2='_baseline'
actual=None
probs=None

def load_data(reform_flag):

    fname = 'simul_agg_det' + str(reform_flag) + '_iceberg.csv' 

    try:
        data = pd.read_csv(inpath + fname)
    except:
        return None
            
    data['y'] = data.y + 1971
    data['tau_nntr'] = data.tau_nntr - 1
    data['tau_applied'] = data.tau_applied-1
    data['iceberg'] = data.iceberg - 1
    data.loc[data.y<1980,'tau_applied'] = data.loc[data.y<1980,'tau_nntr']

    d01 = data.loc[data.y==2001,:].reset_index(drop=True)
    d01['spread'] = np.log((1+d01.tau_nntr)/(1+d01.tau_applied))
    d01 = d01[['i','spread']].drop_duplicates()
    p25 = d01.spread.quantile(0.25)
    p75 = d01.spread.quantile(0.75)
    d01.group=np.nan
    d01.loc[d01.spread<p25,'group']=0
    d01.loc[d01.spread>p75,'group']=1
    data = pd.merge(left=data,right=d01,how='left',on=['i'])

    dlast = data.loc[data.y==data.y.max(),:].reset_index(drop=True)
    dlast.rename(columns={'exports':'exports_last'},inplace=True)
    dlast = dlast[['i','exports_last']].drop_duplicates()
    data = pd.merge(left=data,right=dlast,how='left',on=['i'])
    data['exports2'] = data.exports/data.exports_last
    
    data.sort_values(by=['i','y'],ascending=[True,True],inplace=True)
    data.reset_index(drop=True,inplace=True)

    data['nf_lag'] = data.groupby(['i'])['num_exporters'].transform(lambda x: x.shift())
    data['tau_lag'] = data.groupby(['i'])['tau_applied'].transform(lambda x: x.shift())
    data['tau_lead'] = data.groupby(['i'])['tau_applied'].transform(lambda x: x.shift(-1))

    data['exports_lag'] = data.groupby(['i'])['exports'].transform(lambda x: x.shift())
    data['exports2_lag'] = data.groupby(['i'])['exports2'].transform(lambda x: x.shift())
    data['exports2_lead'] = data.groupby(['i'])['exports2'].transform(lambda x: x.shift(-1))

    data['delta_exports'] = np.log(data.exports) - np.log(data.exports_lag)
    data['delta_tau'] = np.log(1+data.tau_applied) - np.log(1+data.tau_lag)
    data['delta_nf'] = np.log(data.num_exporters) - np.log(data.nf_lag)
    
    return data
       
# load data and aggregate it
df = load_data(0)
df = df[(df.y>=1974)&(df.y<=2008)].reset_index(drop=True)
actual = np.genfromtxt(outpath +'tpu_coeffs.txt')
iceberg = np.genfromtxt(inpath + 'icebergs.txt')

print('\testimating annual NNTR gap coefficients')

#df = df[(df.y>=1974)&(df.y<=2008)].reset_index(drop=True)

formula = 'np.log(exports2) ~ C(y) + C(y):spread'
df2 = df.loc[(df.exports2>1e-8)]

res1 = smf.ols(formula=formula,data=df2).fit(cov_type='HC0')

years = df2.y.unique().tolist()

def effects(res):
    effects_ = []
    ci_ = []
    for y in years:
        effects_.append(res.params['C(y)[%d]:spread'%y])
        ci_.append(res.conf_int(alpha=0.05)[1]['C(y)[%d]:spread'%y])
        
    effects_ = np.asarray(effects_)
    ci_ = np.asarray(ci_)
    ci_ = ci_ - effects_
    
    return effects_, ci_

effects1, ci1 = effects(res1)

_,tmp1 = hpfilter(actual[0:6],lamb=6.25)
_,tmp2 = hpfilter(actual[7:],lamb=6.25)
actual2 = np.append(np.append(tmp1[0:6],actual[6:8]),tmp2[1:])

simulated = np.zeros((len(actual)))

for y in range(1974,2009):
    simulated[y-1974] = res1.params['C(y)[%d]:spread'%y]
    
caldata = np.zeros(len(actual))

s = simulated
x=actual2        
caldata = (s-x)

np.savetxt(outpath +'caldata_iceberg.txt',caldata)

print('\tplotting coefficients')

fig,ax = plt.subplots(1,1,figsize=(4,4))
ax.plot(range(1974,2009),actual,color=colors[2],marker='o',markersize=3,alpha=0.8,label='Data')
ax.plot(range(1974,2009),actual2,color=colors[3],linestyle='--',alpha=0.8,label='Data (smoothed)')
ax.plot(years,effects1,color=colors[1],alpha=0.8,label='Model')
ax.legend(loc='lower right',prop={'size':6})
ax.set_xlim(1974,2008)
#ax.set_ylim(-16,2)
#ax.set_yticks([-15,-10,-5,0])
ax.axhline(0,color='black',linestyle='-',linewidth=1,alpha=1,zorder=1)
ax.axvline(NR,color='black',linestyle=':',linewidth=1,alpha=0.7,zorder=2)
ax.axvline(NU,color='black',linestyle=':',linewidth=1,alpha=0.7,zorder=3)
plt.savefig(outpath + 'model_fig_gap_coefficients_iceberg.pdf',bbox_inches='tight')
plt.close('all')

df_fig = pd.DataFrame({'Year':range(1974,2009),
                       'Data':actual,
                       'Data (smoothed)':actual2,
                       'Iceberg model':effects1})
df_fig.to_csv(outpath+'model_fig_gap_coefficients_iceberg.csv',index=False)


fig, ax = plt.subplots(figsize=(16, 10))
ax.tick_params(axis='both', labelsize=18)
ax.yaxis.label.set_size(18)
sns.despine()
lw=3
tw=20

#fig,ax = plt.subplots(1,1,figsize=(4,4))
t = range(1974,2009)
mu = df.groupby('y').iceberg.mean()
mu0 = df.loc[df.group==0].groupby('y').iceberg.mean()
mu1 = df.loc[df.group==1].groupby('y').iceberg.mean()
ax.plot(t,iceberg,color=colors[0],alpha=0.7,label=r'$\chi_t$',linestyle='--',linewidth=lw)
ax.plot(t,mu0,color=colors[1],alpha=0.7,label=r'Mean NTB (low gap)',linestyle='-',linewidth=lw)
ax.plot(t,mu1,color=colors[3],alpha=0.7,label=r'Mean NTB (high gap)',linestyle='-',marker='x',markersize=10,linewidth=lw)
ax.axvline(NR,color='black',linestyle='--',linewidth=1,alpha=0.7)
ax.axvline(NU,color='black',linestyle='--',linewidth=1,alpha=0.7)
ax.axhline(0,color='black',linestyle='-',linewidth=1,alpha=0.7)
ax.set_xlim(1974,2009)

ax.legend(loc='upper right',prop={'size':tw}) 
fig.subplots_adjust(hspace=0.2,wspace=0.25)
plt.savefig(outpath + 'model_fig_iceberg_vals.pdf',bbox_inches='tight')
plt.close('all')

#df_fig = pd.DataFrame({'Year':t,
#                       'Pr(NNTR to MFN)':p1*np.ones(len(t)),
#                       'Pr(MFN to NNTR)':p2})
#df_fig.to_csv(outpath+'model_fig_probabilities'+suff+'.csv',index=False)

