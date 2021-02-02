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

alpha=0.8
colors=['#377eb8','#e41a1c','#4daf4a','#984ea3']

NT = 100
NU = 21

agg_fns={'f':[('nf',lambda x:x.nunique())],
         'v':[('exports',lambda x:x.sum())]}

def reset_multiindex(df,n,suff):
    levels=df.columns.levels
    labels=df.columns.labels
    df.columns=levels[0][labels[0][0:n]].tolist()+[s+suff for s in levels[1][labels[1][n:]].tolist()]
    return df

def pct_chg(x):
        return 100*(x/x.iloc[0]-1.0)

def wavg(group, avg_name, weight_name):
    d = group[avg_name]
    w = group[weight_name]
    return (d * w).sum() / w.sum()

##################################################
# load data
print('loading data')

def load_data(reform_flag=0):

    fname=''
    if(reform_flag==0):
        fname = 'simul_agg_no_tpu.csv' 
    elif(reform_flag==1):
        fname = 'simul_agg_tpu.csv' 
    elif(reform_flag==2):
        fname = 'simul_agg_tpu2.csv' 

    #print('\nReading data from %s' % fname)
    data = pd.read_csv('../model/output/'+fname)

    #print('Aggregating to simulation/industry/year level...')
    #data2 = reset_multiindex(data.groupby(['s','i','y','tau','tpu_exposure']).agg(agg_fns).reset_index(),5,'')

    return data

# load data and aggregate it
df = [load_data(0),load_data(1),load_data(2)]

#######################################
# plots
print('plotting transition dynamics')

# compute weighted average tariff change
#base_year = df[0].loc[df[0].y==0,:]
#tau_chg = base_year.groupby('s').apply(wavg,'tpu_exposure','exports').mean()

# sum across industries for each simulation
df2 = [x.groupby(['s','y'])[['exports','num_exporters']].sum().reset_index() for x in df]

# average across simulations
df3 = [x.groupby(['y'])[['num_exporters','exports']].mean().reset_index() for x in df2]
for x in df3:
    x['nf_pct_chg'] = x['num_exporters'].transform(pct_chg)
    x['exports_pct_chg'] = x['exports'].transform(pct_chg)

fig,axes = plt.subplots(1,2,figsize=(6.5,3.5),sharex=True,sharey=False)

axes[0].plot(df3[0].y+1979,df3[0].exports_pct_chg,color=colors[0],alpha=0.7,label='No TPU')
axes[0].plot(df3[1].y+1979,df3[1].exports_pct_chg,color=colors[1],alpha=0.7,label='Permanent TPU')
axes[0].plot(df3[2].y+1979,df3[2].exports_pct_chg,color=colors[2],alpha=0.7,label='Temporary TPU')

axes[1].plot(df3[0].y+1979,df3[0].nf_pct_chg,color=colors[0],alpha=0.7)
axes[1].plot(df3[1].y+1979,df3[1].nf_pct_chg,color=colors[1],alpha=0.7)
axes[1].plot(df3[2].y+1979,df3[2].nf_pct_chg,color=colors[2],alpha=0.7)

axes[0].set_title('(a) Exports (\% chg.)')
axes[1].set_title('(b) Num. exporters (\% chg.)')
axes[0].legend(loc='best',prop={'size':8})
axes[0].set_xlim(1979,2010)
axes[0].axvline(1980+NU,color='black',linestyle='--',linewidth=1,alpha=0.7)
axes[1].axvline(1980+NU,color='black',linestyle='--',linewidth=1,alpha=0.7)

fig.subplots_adjust(hspace=0.2,wspace=0.25)
plt.savefig('simul_trans.pdf',bbox='tight')
plt.close('all')

#######################################
# regressions
print('estimating TPU effect')

df2 = [x.groupby(['i','y','tpu_exposure'])[['num_exporters','exports']].mean().reset_index() for x in df]
#df2 = [x.loc[x.y>10,:].reset_index(drop=True) for x in df2]

# naive regression with no control for trade adj dynamics -------------------

formula = 'np.log(exports) ~ C(y) + C(y):np.log(tpu_exposure)'
res = [smf.ols(formula=formula,data=df_).fit(cov_type='HC0') for df_ in df2]

years = df2[1].y.unique().tolist()
years2 = [y+1979 for y in years]

effects = [[],[],[]]

for y in years:
    for i in range(3):
        effects[i].append(res[i].params['C(y)[%d]:np.log(tpu_exposure)'%y])

for i in range(3):
    effects[i] = np.asarray(effects[i])

fig,axes = plt.subplots(1,1,figsize=(4,4),sharex=True,sharey=False)

axes.plot(years2,effects[0],color=colors[0],alpha=0.8,label='No TPU')
axes.plot(years2,effects[1],color=colors[1],alpha=0.8,label='Permanent TPU')
axes.plot(years2,effects[2],color=colors[2],alpha=0.8,label='Temporary TPU')

axes.set_title('TPU effect')
axes.legend(loc='best',prop={'size':8})
axes.set_xlim(1979,2010)
axes.set_xticks([1990,1995,2000,2005,2010])

fig.subplots_adjust(hspace=0.2,wspace=0.25)
plt.savefig('tpu_reg_1980-2010.pdf',bbox='tight')

axes.set_xlim(1990,2010)
axes.set_ylim(-2,0.5)
plt.savefig('tpu_reg_1990-2010.pdf',bbox='tight')


plt.close('all')


# for x in df2:
#     x.sort_values(by=['i','y'],ascending=True,inplace=True)
#     x['l_exports'] = x.groupby(['i'])['exports'].transform(lambda x: x.shift())
#     x['l_num_firms'] = x.groupby(['i'])['exports'].transform(lambda x: x.shift())

# formula = 'np.log(exports) ~ C(y):np.log(tpu_exposure) + C(i) + C(y)'
# res = [smf.ols(formula=formula,data=df_).fit(cov_type='HC0') for df_ in df2]

# effects = [[],[],[]]

# for y in years:
#     for i in range(3):
#         effects[i].append(res[i].params['C(y)[%d]:np.log(tpu_exposure)'%y])

# for i in range(3):
#     effects[i] = np.asarray(effects[i])

# fig,axes = plt.subplots(1,1,figsize=(6.5,3.5),sharex=True,sharey=False)

# axes.plot(years+1980,effects[0],color=colors[0],alpha=0.8,label='No TPU')
# axes.plot(years+1980,effects[1],color=colors[1],alpha=0.8,label='Permanent TPU')
# axes.plot(years+1980,effects[2],color=colors[2],alpha=0.8,label='Temporary TPU')

# axes.set_title('TPU effect w control for num. exporters')
# axes.legend(loc='best',prop={'size':8})
# axes.set_xlim(1990,2010)


# fig.subplots_adjust(hspace=0.2,wspace=0.25)
# plt.savefig('tpu_reg2.pdf',bbox='tight')
# plt.close('all')



