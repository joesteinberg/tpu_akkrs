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
colors=['#377eb8','#e41a1c','#984ea3','#4daf4a']

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

fnames = ['simul_agg_det0_baseline.csv','simul_agg_det0_sunkcost.csv',
          'simul_agg_det0_calvo.csv','simul_agg_tpu_markov_baseline.csv',
          'simul_agg_tpu_markov_sunkcost.csv','simul_agg_tpu_markov_calvo.csv']
slabs = ['No TPU (baseline)','No TPU (sunk cost)','No TPU (Calvo)',
             'TPU (baseline)','TPU (sunk cost)','TPU (Calvo)']

##################################################
# load data

def load_data(fname):

    data = None
    try:
        data = pd.read_csv(inpath + fname)
    except:
        return None

    data['y'] = data.y + 1971
    data['tau_nntr'] = data.tau_nntr - 1
    data['tau_applied'] = data.tau_applied-1
    data.loc[data.y<1980,'tau_applied'] = data.loc[data.y<1980,'tau_nntr']

    d01 = data.loc[data.y==2001,:].reset_index(drop=True)
    d01['spread'] = np.log((1+d01.tau_nntr)/(1+d01.tau_applied))
    d01 = d01[['i','spread']].drop_duplicates()
    data = pd.merge(left=data,right=d01,how='left',on=['i'])

    #data['spread'] = np.log(1+data.gap)

    dlast = data.loc[data.y==data.y.max(),:].reset_index(drop=True)
    dlast.rename(columns={'exports':'exports_last'},inplace=True)
    dlast = dlast[['i','exports_last']].drop_duplicates()
    data = pd.merge(left=data,right=dlast,how='left',on=['i'])
    data['exports2'] = data.exports/data.exports_last
    
    data.sort_values(by=['i','y'],ascending=[True,True],inplace=True)
    data.reset_index(drop=True,inplace=True)

    data['nf_lag'] = data.groupby(['i'])['num_exporters'].transform(lambda x: x.shift())
    data['tau_lag'] = data.groupby(['i'])['tau_applied'].transform(lambda x: x.shift())
    data['exports_lag'] = data.groupby(['i'])['exports'].transform(lambda x: x.shift())
    data['exports2_lag'] = data.groupby(['i'])['exports2'].transform(lambda x: x.shift())
    data['delta_exports'] = np.log(data.exports) - np.log(data.exports_lag)
    data['delta_tau'] = np.log(1+data.tau_applied) - np.log(1+data.tau_lag)
    data['delta_nf'] = np.log(data.num_exporters) - np.log(data.nf_lag)
    data['te'] = data['delta_exports']/(np.log((1+data.tau_applied)/(1+data.tau_lag)))
    
    return data

df = [load_data(fname) for fname in fnames]

#####################################################

# sum.average across industries

df2a = [x.groupby(['y'])[['exports','num_exporters']].sum().reset_index() for x in df]
df2b = [x.groupby(['y'])['tau_applied'].mean().reset_index() for x in df]
df2 = [pd.merge(left=a,right=b,how='left',on=['y']) for a,b in zip(df2a,df2b)]

for x in df2:
    x['nf_pct_chg'] = x['num_exporters'].transform(pct_chg)
    x['exports_pct_chg'] = x['exports'].transform(pct_chg)
    x['nf_growth'] = x['num_exporters'].transform(growth)
    x['exports_growth'] = x['exports'].transform(growth)
 
fig,axes = plt.subplots(1,1,figsize=(4,4))

lns=[]
for i in range(3):
    ln = axes.plot(df2[i].y,np.log(df2[i].exports_pct_chg),color=colors[i],alpha=0.7,label=slabs[i])
    lns=lns+ln

for i in range(3):
    ln = axes.plot(df2[3+i].y,np.log(df2[3+i].exports_pct_chg),color=colors[i],linestyle='--',alpha=0.7,label=slabs[3+i])
    lns=lns+ln

ax2=axes.twinx()
ln = ax2.plot(df2[0].y,df2[0].tau_applied,color=colors[3],alpha=0.7,label='Avg. tariff (right scale)')
lns=lns+ln

axes.axvline(NR,color='black',linestyle='--',linewidth=1,alpha=0.7)
axes.axvline(NU,color='black',linestyle='--',linewidth=1,alpha=0.7)
axes.set_xlim(1974,2008)
axes.set_ylim(-0.1,2)
axes.set_ylabel('Log exports (1974=0)')
ax2.set_ylabel('Avg. tariff')

labs = [l.get_label() for l in lns]
axes.legend(lns,labs,loc='lower right',prop={'size':6})
fig.subplots_adjust(hspace=0.2,wspace=0.25)
plt.savefig(outpath + 'model_fig_agg_trade_sensitivity_models.pdf',bbox_inches='tight')

time = lns[0].get_xdata()
ex_base_notpu = lns[0].get_ydata()
ex_sunk_notpu = lns[1].get_ydata()
ex_calvo_notpu = lns[2].get_ydata()
ex_base_tpu = lns[3].get_ydata()
ex_sunk_tpu = lns[4].get_ydata()
ex_calvo_tpu = lns[5].get_ydata()
tar = lns[-1].get_ydata()
df_fig = pd.DataFrame({'Year':time,
                       'Agg. exports, baseline, no TPU (1974=1)':ex_base_notpu,
                       'Agg. exports, sunk cost, no TPU (1974=1)':ex_sunk_notpu,
                       'Agg. exports, Calvo, no TPU (1974=1)':ex_calvo_notpu,
                       'Agg. exports, baseline, TPU (1974=1)':ex_base_tpu,
                       'Agg. exports, sunk cost,TPU (1974=1)':ex_sunk_tpu,
                       'Agg. exports, Calvo, TPU (1974=1)':ex_calvo_tpu,
                       'Avg. tariff':tar})
df_fig.to_csv('model_fig_agg_trade_sensitivity_models.csv',index=False)

plt.close('all')


###################################################

df = [df_[(df_.y>=1974)&(df_.y<=2008)].reset_index(drop=True) for df_ in df]

df2 = [df_.loc[(df_.exports2>1e-8)] for df_ in df]
formula = 'np.log(exports2) ~ C(y) + C(y):spread'
res1 = [smf.ols(formula=formula,data=df_).fit(cov_type='HC0') for df_ in df2]

years = df2[0].y.unique().tolist()

def effects(res):
    effects_ = [[] for x in df]
    ci_ = [[] for x in df]
    for y in years:
        for i in range(len(df)):
            effects_[i].append(res[i].params['C(y)[%d]:spread'%y])
            ci_[i].append(res[i].conf_int(alpha=0.05)[1]['C(y)[%d]:spread'%y])
    
    for i in range(len(df)):
        effects_[i] = np.asarray(effects_[i])
        ci_[i] = np.asarray(ci_[i])
        ci_[i] = ci_[i] - effects_[i]

    return effects_, ci_

effects1, ci1 = effects(res1)

actual = np.genfromtxt(outpath +'tpu_coeffs.txt')

fig,ax = plt.subplots(1,1,figsize=(4,4))

ax.plot(range(1974,2009),actual,color=colors[3],marker='o',markersize=3,alpha=0.8,label='Data')

for i in range(3):
    ax.plot(years,effects1[i],color=colors[i],alpha=0.8,label=slabs[i])
for i in range(3):
    ax.plot(years,effects1[3+i],color=colors[i],alpha=0.8,label=slabs[3+i],linestyle='--')

ax.legend(loc='lower right',prop={'size':6})
ax.set_xlim(1974,2008)
ax.set_ylim(-16,2)
ax.set_yticks([-15,-10,-5,0])
ax.axhline(0,color='black',linestyle='-',linewidth=1,alpha=1,zorder=1)
ax.axvline(NR,color='black',linestyle=':',linewidth=1,alpha=0.7,zorder=2)
ax.axvline(NU,color='black',linestyle=':',linewidth=1,alpha=0.7,zorder=3)
plt.savefig(outpath + 'model_fig_gap_coefficients_sensitivity_models.pdf',bbox_inches='tight')
plt.close('all')

df_fig = pd.DataFrame({'Year':range(1974,2009),
                       'Data':actual,
                       'Baseline, no TPU':effects1[0],
                       'Sunk cost, no TPU':effects1[1],
                       'Calvo, no TPU':effects1[2],
                       'Baseline, TPU':effects1[3],
                       'Sunk cost, TPU':effects1[4],
                       'Calvo, TPU':effects1[5]})
df_fig.to_csv('model_fig_gap_coefficients_sensitivity_models.csv',index=False)
