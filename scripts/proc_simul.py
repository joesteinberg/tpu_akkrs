#################################################################
# imports, etc.

import numpy as np
import pandas as pd
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf

mpl.rc('text', usetex=True)
mpl.rc('savefig',bbox='tight')
mpl.rc('savefig',format='pdf')
mpl.rc('font',**{'family':'serif','serif':['Palatino'],'size':8})
mpl.rc('font',size=8)
mpl.rc('lines',linewidth=1)

##################################################
# constants and utility functions

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
        return (x/x.iloc[0])
    
def growth(x):
        return 100*(x/x.shift()-1.0)

def wavg(group, avg_name, weight_name):
    d = group[avg_name]
    w = group[weight_name]
    return (d * w).sum() / w.sum()

##################################################
# load data
print('\tloading data')

def load_data(reform_flag=0):

    fname=''
    if(reform_flag==0):
        fname = 'simul_agg_no_tpu.csv' 
    elif(reform_flag==1):
        fname = 'simul_agg_tpu_temp.csv' 
    elif(reform_flag==2):
        fname = 'simul_agg_tpu_perm.csv' 

    #print('\nReading data from %s' % fname)
    data = pd.read_csv(inpath + fname)

    #data['cnt'] = data.groupby('i')['exports'].transform(lambda x: (x>1e-8).sum())
    #data=data[data.cnt==(len(data.y.unique())*len(data.s.unique()))].reset_index(drop=True)
    
    data['y'] = data.y + 1971
    data['tau_nntr'] = data.tau_nntr - 1
    data['tau_applied'] = data.tau_applied-1

    d01 = data.loc[data.y==2001,:].reset_index(drop=True)
    d01['spread'] = np.log((1+d01.tau_nntr)/(1+d01.tau_applied))
    d01 = d01[['i','spread']].drop_duplicates()
    data = pd.merge(left=data,right=d01,how='left',on=['i'])

    dlast = data.loc[data.y==data.y.max(),:].reset_index(drop=True)
    dlast.rename(columns={'exports':'exports_last'},inplace=True)
    dlast = dlast[['s','i','exports_last']].drop_duplicates()
    data = pd.merge(left=data,right=dlast,how='left',on=['s','i'])
    data['exports2'] = data.exports/data.exports_last
    
    data.sort_values(by=['s','i','y'],ascending=[True,True,True],inplace=True)
    data.reset_index(drop=True,inplace=True)

    data['exports_lag'] = data.groupby(['s','i'])['exports'].transform(lambda x: x.shift())
    data['exports2_lag'] = data.groupby(['s','i'])['exports2'].transform(lambda x: x.shift())
    data = data[(data.y>=1974)&(data.y<=2008)].reset_index(drop=True)
    
    return data

# load data and aggregate it
df = [load_data(0),load_data(1),load_data(2)]
probs = np.genfromtxt(inpath + 'tpuprobs.txt')

#######################################
# plots
print('\tplotting transition dynamics')

# sum across industries for each simulation
df2a = [x.groupby(['s','y'])[['exports','num_exporters','exits','entries']].sum().reset_index() for x in df]

df2b = [x.groupby(['s','y'])['tau_applied'].mean().reset_index() for x in df]

df2 = [pd.merge(left=a,right=b,how='left',on=['s','y']) for a,b in zip(df2a,df2b)]

# average across simulations
df3 = [x.groupby(['y'])[['num_exporters','exports','tau_applied']].mean().reset_index() for x in df2]
for x in df3:
    x['nf_pct_chg'] = x['num_exporters'].transform(pct_chg)
    x['exports_pct_chg'] = x['exports'].transform(pct_chg)
    x['nf_growth'] = x['num_exporters'].transform(growth)
    x['exports_growth'] = x['exports'].transform(growth)
    
fig,axes = plt.subplots(1,1,figsize=(4,4))

ln1 = axes.plot(df3[0].y,np.log(df3[0].exports_pct_chg),color=colors[0],alpha=0.7,label='No TPU')
ln2 = axes.plot(df3[1].y,np.log(df3[1].exports_pct_chg),color=colors[1],alpha=0.7,label='Temporary TPU')
ln3 = axes.plot(df3[2].y,np.log(df3[2].exports_pct_chg),color=colors[2],alpha=0.7,label='Permanent TPU')

ax2=axes.twinx()
#ln4 = ax2.plot(range(1980,2001),probs[0],linestyle='--',color=colors[1],alpha=0.7,label='Temporary TPU probs')
#ln5 = ax2.plot(range(1980,2001),probs[1],linestyle='--',color=colors[2],alpha=0.7,label='Permanent TPU probs')
ln4 = ax2.plot(df3[0].y,df3[0].tau_applied,color=colors[3],alpha=0.7,label='Avg. tariff (right scale)')

axes.axvline(NR,color='black',linestyle='--',linewidth=1,alpha=0.7)
axes.axvline(NU,color='black',linestyle='--',linewidth=1,alpha=0.7)
axes.set_xlim(1974,2008)
axes.set_ylim(0,3.5)
axes.set_ylabel('Log exports (1974 = 0)')
ax2.set_ylabel('Avg. tariff')
axes.set_title('Trade growth vs. applied tariffs')

lns = ln1+ln2+ln3+ln4
labs = [l.get_label() for l in lns]
axes.legend(lns,labs,loc='best',prop={'size':6})
fig.subplots_adjust(hspace=0.2,wspace=0.25)
plt.savefig(outpath + 'simul_trans.pdf',bbox_inches='tight')
plt.close('all')

#######################################
# regressions
print('\testimating TPU effect')

df2 = [df_.loc[(df_.exports2>1e-8)] for df_ in df]
formula = 'np.log(exports2) ~ C(y) + C(y):spread'
res1 = [smf.ols(formula=formula,data=df_).fit(cov_type='HC0') for df_ in df2]

df2 = [df_.loc[(df_.exports2>1e-8) & (df_.exports2_lag>1e-8)] for df_ in df2]
formula = 'np.log(exports2) ~ C(y) + C(y):spread + np.log(exports2_lag)'
res2 = [smf.ols(formula=formula,data=df_).fit(cov_type='HC0') for df_ in df2]

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
effects2, ci2 = effects(res2)

actual = np.genfromtxt(outpath +'tpu_coeffs.txt')

simulated = [np.zeros((2001-1980+1)) for x in df[1:]]

for i in range(len(df[1:])):
    for y in range(1980,2002):
        simulated[i][y-1980] = res1[i+1].params['C(y)[%d]:spread'%y]

caldata = np.zeros((len(df[1:]),2001-1981+1))
for i in range(len(df[1:])):
    s = simulated[i]
    x=actual[1:]*(s[0]/actual[0])
    #caldata[i,:] = (s[1:]-x)/np.abs(x)
    caldata[i,:] = (s[1:]-x)

np.savetxt(outpath +'caldata.txt',caldata)

print('\tplotting coefficients')
fig,axes = plt.subplots(1,2,figsize=(6.5,3.5),sharex=False,sharey=False)

ln1=axes[0].plot(range(1980,2002),actual*(simulated[0][0]/actual[0]),color=colors[3],marker='o',markersize=3,alpha=0.8,label='Data')

ln3=axes[0].plot(years,effects1[1],color=colors[1],alpha=0.8,label='Temporary TPU')
#axes[0].fill_between(years, (effects1[1]-ci1[1]), (effects1[1]+ci1[1]), color=colors[1], alpha=.1)

ln4=axes[0].plot(years,effects1[2],color=colors[2],alpha=0.8,label='Permanent TPU')
#axes[0].fill_between(years, (effects1[2]-ci1[2]), (effects1[2]+ci1[2]), color=colors[2], alpha=.1)

ln2=axes[0].plot(years,effects1[0],color=colors[0],alpha=0.8,label='No TPU')
#axes[0].fill_between(years, (effects1[0]-ci1[0]), (effects1[0]+ci1[0]), color=colors[0], alpha=.1)


lns = ln1+ln2+ln3+ln4
labs = [l.get_label() for l in lns]
axes[0].legend(lns,labs,loc='best',prop={'size':6})
axes[0].set_title('(a) NNTR gap coefficients')
axes[0].set_xlim(1974,2008)
axes[0].axvline(NR,color='black',linestyle=':',linewidth=1,alpha=0.7)
axes[0].axvline(NU,color='black',linestyle=':',linewidth=1,alpha=0.7)

#axes[1].plot(years,effects2[0],color=colors[0],alpha=0.8,label='No TPU')
#axes[1].fill_between(years, (effects2[0]-ci2[0]), (effects2[0]+ci2[0]), color=colors[0], alpha=.1)

#axes[1].plot(years,effects2[1],color=colors[1],alpha=0.8,label='Temporary TPU')
#axes[1].fill_between(years, (effects2[1]-ci2[1]), (effects2[1]+ci2[1]), color=colors[1], alpha=.1)

#axes[1].plot(years,effects2[2],color=colors[2],alpha=0.8,label='Permanent TPU')
#axes[1].fill_between(years, (effects2[2]-ci2[2]), (effects2[2]+ci2[2]), color=colors[2], alpha=.1)

axes[1].plot(range(1980,2001),probs[0],color=colors[1],alpha=0.7,label='Temporary TPU')
axes[1].plot(range(1980,2001),probs[1],color=colors[2],alpha=0.7,label='Permanent TPU')

axes[1].set_title('(b) Reversion probabilities')
axes[1].set_xlim(1980,2000)

fig.subplots_adjust(hspace=0.2,wspace=0.25)
plt.savefig(outpath + 'tpu_regs.pdf',bbox_inches='tight')
plt.close('all')

#######################################
# regressions

