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

if (len(sys.argv)==1 or (len(sys.argv)>1 and sys.argv[1]=='-calc_te')):
    actual = np.genfromtxt(outpath +'tpu_coeffs.txt')
    probs = np.genfromtxt(inpath + 'tpuprobs_markov_baseline.txt')
elif(len(sys.argv)>1 and sys.argv[1]=='-permtpu'):
    actual = np.genfromtxt(outpath +'tpu_coeffs.txt')
    probs = np.genfromtxt(inpath + 'tpuprobs_markov_permtpu.txt')
    suff='_permtpu'
    suff2='_permtpu'
elif(len(sys.argv)>1 and sys.argv[1]=='-tsusa'):
    actual = np.genfromtxt(outpath +'tpu_coeffs_alt1_tsusa.txt')
    probs = np.genfromtxt(inpath + 'tpuprobs_markov_tsusa.txt')
    suff='_alt_coeffs_tsusa'
    suff2='_tsusa'
elif(len(sys.argv)>1 and sys.argv[1]=='-inv'):
    actual = np.genfromtxt(outpath +'tpu_coeffs_alt2_inv.txt')
    probs = np.genfromtxt(inpath + 'tpuprobs_markov_inv.txt')
    suff='_alt_coeffs_inv'
    suff2='_inv'
elif(len(sys.argv)>1 and sys.argv[1]=='-sitc68'):
    actual = np.genfromtxt(outpath +'tpu_coeffs_sitc68.txt')
    probs = np.genfromtxt(inpath + 'tpuprobs_markov_sitc68.txt')
    suff='_sitc68'
    suff2='_sitc68'
elif(len(sys.argv)>1 and sys.argv[1]=='-sitc7'):
    actual = np.genfromtxt(outpath +'tpu_coeffs_sitc7.txt')
    probs = np.genfromtxt(inpath + 'tpuprobs_markov_sitc7.txt')
    suff='_sitc7'
    suff2='_sitc7'
elif(len(sys.argv)>1 and sys.argv[1]=='-ci_lower'):
    actual = np.genfromtxt(outpath +'tpu_coeffs_ci_lower.txt')
    probs = np.genfromtxt(inpath + 'tpuprobs_markov_ci_lower.txt')
    suff='_ci_lower'
    suff2='_ci_lower'
elif(len(sys.argv)>1 and sys.argv[1]=='-ci_upper'):
    actual = np.genfromtxt(outpath +'tpu_coeffs_ci_upper.txt')
    probs = np.genfromtxt(inpath + 'tpuprobs_markov_ci_upper.txt')
    suff='_ci_upper'
    suff2='_ci_upper'

    
def load_data(reform_flag=0):

    fname=''
    if(reform_flag==0):
        fname = 'simul_agg_det0'+suff2+'.csv' 
    elif(reform_flag==3):
        fname = 'simul_agg_tpu_markov'+suff2+'.csv' 

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

    data['industry'] = data['i'].astype(str).str[0:1]
    if(suff=='_sitc7'):
        data = data[data.industry=='7']
        data.reset_index(inplace=True,drop=True)
    elif(suff=='_sitc68'):
        data = data[(data.industry=='6') | (data.industry=='8')]
        data.reset_index(inplace=True,drop=True)
    
    return data
       
# load data and aggregate it
df = [load_data(0),load_data(3)]


#######################################
# plots
print('\tplotting transition dynamics')

# average across simulations
df2a = [x.groupby(['y'])[['exports','num_exporters']].sum().reset_index() for x in df]
df2b = [x.groupby(['y'])[['tau_applied','pv_tau']].mean().reset_index() for x in df]
df2 = [pd.merge(left=a,right=b,how='left',on=['y']) for a,b in zip(df2a,df2b)]

for x in df2:
    x['nf_pct_chg'] = x['num_exporters'].transform(pct_chg)
    x['exports_pct_chg'] = x['exports'].transform(pct_chg)
    x['nf_growth'] = x['num_exporters'].transform(growth)
    x['exports_growth'] = x['exports'].transform(growth)
    
fig,axes = plt.subplots(1,1,figsize=(4,4))

ln1 = axes.plot(df2[0].y,np.log(df2[0].exports_pct_chg),color=colors[0],alpha=0.7,label='No TPU')
ln2 = axes.plot(df2[1].y,np.log(df2[1].exports_pct_chg),color=colors[1],alpha=0.7,label='TPU')

ax2=axes.twinx()
ln3 = ax2.plot(df2[0].y,df2[0].tau_applied,color=colors[2],alpha=0.7,label='Avg. tariff (right scale)')

axes.axvline(NR,color='black',linestyle='--',linewidth=1,alpha=0.7)
axes.axvline(NU,color='black',linestyle='--',linewidth=1,alpha=0.7)
axes.set_xlim(1974,2008)
axes.set_ylim(-0.1,2)
axes.set_ylabel('Log exports (1974=0)')
ax2.set_ylabel('Avg. tariff')

lns = ln1+ln2+ln3
labs = [l.get_label() for l in lns]
axes.legend(lns,labs,loc='lower right',prop={'size':6})
fig.subplots_adjust(hspace=0.2,wspace=0.25)
plt.savefig(outpath + 'model_fig_agg_trade'+suff+'.pdf',bbox_inches='tight')

time = ln1[0].get_xdata()
ex_notpu = ln1[0].get_ydata()
ex_tpu = ln2[0].get_ydata()
tar = ln3[0].get_ydata()
df_fig = pd.DataFrame({'Year':time,
                       'Agg. exports, no TPU (1974=1)':ex_notpu,
                       'Agg. exports, TPU (1974=1)':ex_tpu,
                       'Avg. tariff':tar})
df_fig.to_csv(outpath+'model_fig_agg_trade'+suff+'.csv',index=False)

plt.close('all')


fig,axes = plt.subplots(1,1,figsize=(4,4))

#ln1 = axes.plot(df2[0].y,df2[0].tau_applied,color=colors[2],alpha=0.7,label='Mean')

ln2 = axes.plot(df2[0].y,df2[0].pv_tau-1,color=colors[0],alpha=0.7,label='PV (No TPU)')
ln3 = axes.plot(df2[1].y,df2[1].pv_tau-1,color=colors[1],alpha=0.7,label='PV (TPU)')


axes.axvline(NR,color='black',linestyle='--',linewidth=1,alpha=0.7)
axes.axvline(NU,color='black',linestyle='--',linewidth=1,alpha=0.7)
axes.set_xlim(1974,2008)
axes.set_ylim(0,0.35)
#axes.set_ylabel('PV tariff')
#ax2.set_ylabel('Avg. tariff')

lns = ln1+ln2+ln3
labs = [l.get_label() for l in lns]
axes.legend(lns,labs,loc='lower right',prop={'size':6})
fig.subplots_adjust(hspace=0.2,wspace=0.25)
plt.savefig(outpath + 'model_fig_pv_tariff'+suff+'.pdf',bbox_inches='tight')

time = ln1[0].get_xdata()
pv_notpu = ln1[0].get_ydata()
pv_tpu = ln2[0].get_ydata()
tar = ln3[0].get_ydata()
df_fig = pd.DataFrame({'Year':time,
                       'PV tariff no TPU (1974=1)':pv_notpu,
                       'PV tariff TPU (1974=1)':pv_tpu,
                       'Avg. tariff':tar})
df_fig.to_csv(outpath+'model_fig_pv_tariff'+suff+'.csv',index=False)

plt.close('all')


#######################################
# regressions

#df = [df_[(df_.y>=1974)&(df_.y<=2008)].reset_index(drop=True) for df_ in df]

#if len(sys.argv)>1 and sys.argv[1]=='-calc_te':
if(len(sys.argv)>1 and '-calc_te' in sys.argv):
    
    print('\testimating trade elasticities via ECM')
    print(df[0].y.max())

    SR_true = -2.3
    LR_true = -8.07

    if(suff2=='_sitc68'):
        SR_true = -2.41
        LR_true = -7.32
    elif(suff2=='_sitc7'):
        SR_true = -3.75
        LR_true = -17.61

    df2 = [df_.loc[(df_.exports>1e-8) & (df_.exports_lag>1.0e-8)] for df_ in df]
    formula = 'delta_exports ~ np.log(1+tau_lag) + np.log(exports_lag) + delta_tau + C(i)'
    eres3 = smf.ols(formula=formula,data=df2[1]).fit(cov_type='HC0')

    print("\tSR\tLR")
    print("Model  %0.3f\t%0.3f" % (eres3.params['delta_tau'],-eres3.params['np.log(1 + tau_lag)']/eres3.params['np.log(exports_lag)']))
    print("Data   %0.3f\t%0.3f" % (SR_true,LR_true))


#######################################
# regressions
print('\testimating annual NNTR gap coefficients')

df = [df_[(df_.y>=1974)&(df_.y<=2008)].reset_index(drop=True) for df_ in df]
df2=None
formula=''

#if(len(sys.argv)>1 and sys.argv[1]=='-inv'):
#    formula = 'np.log(exports2) ~ C(y) + C(y):spread + np.log(1 + tau_lead)'
#    df2 = [df_.loc[(df_.exports2>1e-8) & (df_.exports2_lead>1e-8)] for df_ in df]
#else:
formula = 'np.log(exports2) ~ C(y) + C(y):spread'
df2 = [df_.loc[(df_.exports2>1e-8)] for df_ in df]
    
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

if(len(sys.argv)>1 and sys.argv[1]=='-inv'):
    _,tmp1 = hpfilter(actual[0:6],lamb=6.25)
    _,tmp2 = hpfilter(actual[7:],lamb=6.25)
    actual2 = np.append(np.append(tmp1,actual[6:8]),tmp2[1:])
else:
    _,tmp = hpfilter(actual[7:],lamb=6.25)
    actual2 = np.append(actual[0:8],tmp[1:])

simulated = [np.zeros((len(actual))) for x in df[1:]]

for i in range(len(df[1:])):
    for y in range(1974,2009):
        simulated[i][y-1974] = res1[i+1].params['C(y)[%d]:spread'%y]

caldata = np.zeros((len(df[1:]),len(actual)))
for i in range(len(df[1:])):
    s = simulated[i]
    x=actual2        
    caldata[i,:] = (s-x)

np.savetxt(outpath +'caldata'+suff2+'.txt',caldata[0,:])

print('\tplotting coefficients')

fig,ax = plt.subplots(1,1,figsize=(4,4))
ax.plot(range(1974,2009),actual,color=colors[2],marker='o',markersize=3,alpha=0.8,label='Data')
#ax.plot(range(1974,2009),actual2,color=colors[3],linestyle='--',alpha=0.8,label='Data (smoothed)')
ax.plot(years,effects1[1],color=colors[1],alpha=0.8,label='TPU')
ax.plot(years,effects1[0],color=colors[0],alpha=0.8,label='No TPU',linestyle='--')
ax.legend(loc='lower right',prop={'size':6})
ax.set_xlim(1974,2008)
#ax.set_ylim(-16,2)
ax.set_yticks([-15,-10,-5,0])
ax.axhline(0,color='black',linestyle='-',linewidth=1,alpha=1,zorder=1)
ax.axvline(NR,color='black',linestyle=':',linewidth=1,alpha=0.7,zorder=2)
ax.axvline(NU,color='black',linestyle=':',linewidth=1,alpha=0.7,zorder=3)
plt.savefig(outpath + 'model_fig_gap_coefficients'+suff+'.pdf',bbox_inches='tight')
plt.close('all')

df_fig = pd.DataFrame({'Year':range(1974,2009),
                       'Data':actual,
                       'Data (smoothed)':actual2,
                       'No TPU':effects1[0],
                       'TPU':effects1[1]})
df_fig.to_csv(outpath+'model_fig_gap_coefficients'+suff+'.csv',index=False)

fig,ax = plt.subplots(1,1,figsize=(4,4))

t = range(1973,2008)
p1 = probs.copy()
p1[NR-1-1973:]=p1[NR-1-1973]
ax.plot(t,p1,color=colors[0],alpha=0.7,label=r'$P(NNTR\rightarrow MFN)$')
p2 = probs.copy()
p2[0:(NR-1973)] = probs[(NR-1973)]
ax.plot(t,p2,color=colors[1],alpha=0.7,label=r'$P(MFN\rightarrow NNTR)$')
ax.axvline(NR,color='black',linestyle=':',linewidth=1,alpha=0.7)
ax.axvline(NU,color='black',linestyle=':',linewidth=1,alpha=0.7)
ax.set_xlim(1973,2008)
ax.legend(loc='upper right',prop={'size':6}) 
fig.subplots_adjust(hspace=0.2,wspace=0.25)
plt.savefig(outpath + 'model_fig_probabilities'+suff+'.pdf',bbox_inches='tight')
plt.close('all')

df_fig = pd.DataFrame({'Year':t,
                       'Pr(NNTR to MFN)':p1*np.ones(len(t)),
                       'Pr(MFN to NNTR)':p2})
df_fig.to_csv(outpath+'model_fig_probabilities'+suff+'.csv',index=False)

