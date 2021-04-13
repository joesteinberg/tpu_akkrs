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
        return (x/x.iloc[8])
    
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
        fname = 'simul_agg_det0.csv' 
    elif(reform_flag==3):
        fname = 'simul_agg_tpu_markov.csv' 

    try:
        data = pd.read_csv(inpath + fname)
    except:
        return None
        

    #data['cnt'] = data.groupby('i')['exports'].transform(lambda x: (x>1e-8).sum())
    #data=data[data.cnt==(len(data.y.unique())*len(data.s.unique()))].reset_index(drop=True)
    
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
    data['exports_lag'] = data.groupby(['i'])['exports'].transform(lambda x: x.shift())
    data['exports2_lag'] = data.groupby(['i'])['exports2'].transform(lambda x: x.shift())
    data['delta_exports'] = np.log(data.exports) - np.log(data.exports_lag)
    data['delta_tau'] = np.log(1+data.tau_applied) - np.log(1+data.tau_lag)
    data['delta_nf'] = np.log(data.num_exporters) - np.log(data.nf_lag)
    data['te'] = data['delta_exports']/(np.log((1+data.tau_applied)/(1+data.tau_lag)))
    #data = data[(data.y>=1974)&(data.y<=2008)].reset_index(drop=True)
    #data = data[(data.y>=1974)&(data.y<=2008)].reset_index(drop=True)
    
    return data
       
# load data and aggregate it
df = [load_data(0),load_data(reform_flag)]

probs = np.genfromtxt(inpath + 'tpuprobs_markov.txt')

#######################################
# plots
print('\tplotting transition dynamics')

# sum across industries for each simulation
df2a = [x.groupby(['y'])[['exports','num_exporters']].sum().reset_index() for x in df]

df2b = [x.groupby(['y'])['tau_applied'].mean().reset_index() for x in df]

df2 = [pd.merge(left=a,right=b,how='left',on=['y']) for a,b in zip(df2a,df2b)]

# average across simulations
for x in df2:
    x['nf_pct_chg'] = x['num_exporters'].transform(pct_chg)
    x['exports_pct_chg'] = x['exports'].transform(pct_chg)
    x['nf_growth'] = x['num_exporters'].transform(growth)
    x['exports_growth'] = x['exports'].transform(growth)
    
fig,axes = plt.subplots(1,1,figsize=(4,4))

ln1 = axes.plot(df2[0].y,np.log(df2[0].exports),color=colors[0],alpha=0.7,label='No TPU')
ln2 = axes.plot(df2[1].y,np.log(df2[1].exports),color=colors[1],alpha=0.7,label='TPU')

ax2=axes.twinx()
ln3 = ax2.plot(df2[0].y,df2[0].tau_applied,color=colors[2],alpha=0.7,label='Avg. tariff (right scale)')

axes.axvline(NR,color='black',linestyle='--',linewidth=1,alpha=0.7)
axes.axvline(NU,color='black',linestyle='--',linewidth=1,alpha=0.7)
axes.set_xlim(1974,2008)
axes.set_ylim(4,6.5)
axes.set_ylabel('Log exports')
ax2.set_ylabel('Avg. tariff')
#axes.set_title('Trade growth vs. applied tariffs')

lns = ln1+ln2+ln3
labs = [l.get_label() for l in lns]
axes.legend(lns,labs,loc='lower right',prop={'size':6})
fig.subplots_adjust(hspace=0.2,wspace=0.25)
plt.savefig(outpath + 'simul_trans.pdf',bbox_inches='tight')
plt.close('all')

#######################################
# regressions

if len(sys.argv)>1 and sys.argv[1]=='calc_te':

    print('\testimating trade elasticities')

    for df_ in df:
        df_['cnt'] = df_.groupby('i')['exports'].transform(lambda x: (x>1e-8).sum())

        df2 = [df_.loc[(df_.exports>1e-8)] for df_ in df]
        formula = 'np.log(exports) ~ np.log(1+tau_applied)'
        eres1 = [smf.ols(formula=formula,data=df_).fit(cov_type='HC0') for df_ in df2]

        df2 = [df_.loc[(df_.exports>1e-8) & (df_.exports_lag>1.0e-8)] for df_ in df]
        formula = 'np.log(exports) ~ np.log(1+tau_applied) + np.log(exports_lag) + C(i)'
        eres2 = [smf.ols(formula=formula,data=df_).fit(cov_type='HC0') for df_ in df2]

        df2 = [df_.loc[(df_.exports>1e-8) & (df_.exports_lag>1.0e-8)] for df_ in df]
        formula = 'delta_exports ~ np.log(1+tau_lag) + np.log(exports_lag) + delta_tau + C(i)'
        eres3 = [smf.ols(formula=formula,data=df_).fit(cov_type='HC0') for df_ in df2]

        te80 = [df_.loc[(df_.y==1980)&(df_.te.notna())&(df_.te>-9999)&(df_.te<9999)][['tau_applied','tau_lag','te']].mean() for df_ in df]

    print("\tSR\tLR")
    print("NoTPU   %0.3f\t%0.3f" % (eres3[0].params['delta_tau'],-eres3[0].params['np.log(1 + tau_lag)']/eres3[0].params['np.log(exports_lag)']))
    print("Markov  %0.3f\t%0.3f" % (eres3[1].params['delta_tau'],-eres3[1].params['np.log(1 + tau_lag)']/eres3[1].params['np.log(exports_lag)']))

    file = open(outpath + 'model_ecm.tex','w')

    # header
    #file.write('\\begin{landscape}\n')
    file.write('\\begin{table}[p]\n')
    file.write('\\footnotesize\n')
    file.write('\\renewcommand{\\arraystretch}{1.2}\n')
    file.write('\\begin{center}\n')
    file.write("\\caption{Short and long-run trade elasticities in simulated data}\n")
    file.write('\\label{tab:model_ecm}\n')
    file.write('\\begin{tabular}{lcccccc')
    file.write('}')
    file.write('\\toprule\n')
    
    colname = lambda s: '\\multicolumn{1}{b{1.5cm}}{\centering '+s+'}'
    # column names
    file.write('& \\multicolumn{3}{c}{No TPU} & \\multicolumn{3}{c}{TPU}\\\\\n')
    file.write('\\cmidrule(rl){2-4}\\cmidrule(rl){5-7}\n')
    file.write('Dep. var.')
    
    for i in range(2):
        file.write('&'+colname(r'Cross-\\section'))
        file.write('&'+colname(r'ECM\\restricted'))
        file.write('&'+colname(r'ECM\\unrestricted'))
    file.write('\\\\\n\\midrule\n')

    # numbers
    file.write('$\\tau_{gt}$')
    for i in range(2):
        file.write('&%0.3f'%eres1[i].params['np.log(1 + tau_applied)'])
        file.write('&%0.3f&'%eres2[i].params['np.log(1 + tau_applied)'])
    file.write('\\\\\n')

    file.write('$\\Delta\\tau_{gt}$')
    for i in range(2):
        file.write('&&&%0.3f'%eres3[i].params['delta_tau'])
    file.write('\\\\\n')
        
    file.write('$v_{g,t-1}$')
    for i in range(2):
        file.write('&&%0.3f'%eres2[i].params['np.log(exports_lag)'])
        file.write('&%0.3f'%eres3[i].params['np.log(exports_lag)'])
    file.write('\\\\\n')
        
    file.write('$\\tau_{g,t-1}$')
    for i in range(2):
        file.write('&&&%0.3f'%eres3[i].params['np.log(1 + tau_lag)'])
    file.write('\\\\\n')

    file.write('\\midrule\n')
    file.write('Long-run')
    for i in range(2):
        file.write('&&%0.3f'%(eres2[i].params['np.log(1 + tau_applied)']/(1-eres2[i].params['np.log(exports_lag)'])))
        file.write('&%0.3f'%(-eres3[i].params['np.log(1 + tau_lag)']/eres3[i].params['np.log(exports_lag)']))
    file.write('\\\\\n')

    file.write('Long/short-run')
    for i in range(2):
        lr = eres2[i].params['np.log(1 + tau_applied)']/(1-eres2[i].params['np.log(exports_lag)'])
        sr = eres2[i].params['np.log(1 + tau_applied)']
        file.write('&&%0.3f'%(lr/sr))
        
        lr = -eres3[i].params['np.log(1 + tau_lag)']/eres3[i].params['np.log(exports_lag)']
        sr = eres3[i].params['delta_tau']
        file.write('&%0.3f'%(lr/sr))
    file.write('\\\\\n')


    # footer
    file.write('\\bottomrule\n')
    file.write('\\end{tabular}\n')
    file.write('\\end{center}\n')
    file.write('\\normalsize\n')
    file.write('\\end{table}\n')
    
    file.close()


#######################################
# regressions
print('\testimating TPU effect')

df = [df_[(df_.y>=1974)&(df_.y<=2008)].reset_index(drop=True) for df_ in df]

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
_,tmp = hpfilter(actual[7:],lamb=6.25)
actual2 = np.append(actual[0:8],tmp[1:])

simulated = [np.zeros((len(actual))) for x in df[1:]]

for i in range(len(df[1:])):
    for y in range(1974,2009):
        simulated[i][y-1974] = res1[i+1].params['C(y)[%d]:spread'%y]

caldata = np.zeros((len(df[1:]),len(actual)))
for i in range(len(df[1:])):
    s = simulated[i]

    #if(reform_flag==2):
    #    x=actual*(s[6]/actual[6])
    #else:
        #x=actual*(s[0:6].mean()/actual[0:6].mean())
    #    x=actual
    x=actual2
        
    caldata[i,:] = (s-x)

np.savetxt(outpath +'caldata.txt',caldata[0,:])

print('\tplotting coefficients')

fig,ax = plt.subplots(1,1,figsize=(4,4))
ax.plot(range(1974,2009),actual,color=colors[2],marker='o',markersize=3,alpha=0.8,label='Data')
ax.plot(range(1974,2009),actual2,color=colors[3],linestyle='--',alpha=0.8,label='Data (smoothed)')
ax.plot(years,effects1[1],color=colors[1],alpha=0.8,label='TPU')
ax.plot(years,effects1[0],color=colors[0],alpha=0.8,label='No TPU')
ax.legend(loc='lower right',prop={'size':6})
#ax.set_title('(a) NNTR gap coefficients')
ax.set_xlim(1974,2008)
ax.axvline(NR,color='black',linestyle=':',linewidth=1,alpha=0.7)
ax.axvline(NU,color='black',linestyle=':',linewidth=1,alpha=0.7)
plt.savefig(outpath + 'tpu_regs.pdf',bbox_inches='tight')
plt.close('all')

fig,ax = plt.subplots(1,1,figsize=(4,4))
t = range(1974,2008)
p1 = probs[0]
ax.plot(t,p1*np.ones(len(t)),color=colors[0],alpha=0.7,label=r'$P(NNTR\rightarrow MFN)$')
p2 = probs
p2[0:(NR-1974)] = probs[(NR-1974)]
ax.plot(t,p2,color=colors[1],alpha=0.7,label=r'$P(MFN\rightarrow NNTR)$')
ax.axvline(NR,color='black',linestyle=':',linewidth=1,alpha=0.7)
ax.axvline(NU,color='black',linestyle=':',linewidth=1,alpha=0.7)
ax.set_xlim(1974,2008)
ax.legend(loc='upper right',prop={'size':6}) 
#ax.set_title('(b) Reversion probabilities')  
fig.subplots_adjust(hspace=0.2,wspace=0.25)
plt.savefig(outpath + 'tpu_probs.pdf',bbox_inches='tight')
plt.close('all')

