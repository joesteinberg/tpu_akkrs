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

def load_data(reform_flag=0):

    fname=''
    if(reform_flag==0):
        fname = 'simul_agg_det0_baseline.csv'
    elif(reform_flag==1):
        fname = 'simul_agg_det1_baseline.csv' 
    elif(reform_flag==2):
        fname = 'simul_agg_det2_baseline.csv'
    elif(reform_flag==4):
        fname = 'simul_agg_det4_baseline.csv' 
    elif(reform_flag==3):
        fname = 'simul_agg_tpu_markov_baseline.csv' 

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
    
    return data
       
# load data and aggregate it
df = [load_data(0),load_data(1),load_data(4),load_data(3),load_data(2)]


#######################################
# regressions


for df_ in df:
    df_['cnt'] = df_.groupby('i')['exports'].transform(lambda x: (x>1e-8).sum())

#df2 = [df_.loc[(df_.exports>1e-8)] for df_ in df]
#formula = 'np.log(exports) ~ np.log(1+tau_applied)'
#eres1 = [smf.ols(formula=formula,data=df_).fit(cov_type='HC0') for df_ in df2]

print('\testimating cross sectional regression')

df2b = [df_.loc[(df_.exports>1e-8) & (df_.y==2070)] for df_ in df]
formula = 'np.log(exports) ~ np.log(1+tau_applied)'
eres1b = [smf.ols(formula=formula,data=df_).fit(cov_type='HC0') for df_ in df2b]

#df2 = [df_.loc[(df_.exports>1e-8) & (df_.exports_lag>1.0e-8)] for df_ in df]
#formula = 'np.log(exports) ~ np.log(1+tau_applied) + np.log(exports_lag) + C(i)'
#eres2 = [smf.ols(formula=formula,data=df_).fit(cov_type='HC0') for df_ in df2]

print('\testimating unrestricted ECM')

df2 = [df_.loc[(df_.exports>1e-8) & (df_.exports_lag>1.0e-8)] for df_ in df]
formula = 'delta_exports ~ np.log(1+tau_lag) + np.log(exports_lag) + delta_tau + C(i)'
eres3 = [smf.ols(formula=formula,data=df_).fit(cov_type='HC0') for df_ in df2]

print("\tSR\tLR")
print("No TPU 0   %0.3f\t%0.3f" % (eres3[0].params['delta_tau'],-eres3[0].params['np.log(1 + tau_lag)']/eres3[0].params['np.log(exports_lag)']))
print("No TPU 2   %0.3f\t%0.3f" % (eres3[1].params['delta_tau'],-eres3[1].params['np.log(1 + tau_lag)']/eres3[1].params['np.log(exports_lag)']))
print("No TPU 4   %0.3f\t%0.3f" % (eres3[2].params['delta_tau'],-eres3[2].params['np.log(1 + tau_lag)']/eres3[2].params['np.log(exports_lag)']))
print("Markov  %0.3f\t%0.3f" % (eres3[3].params['delta_tau'],-eres3[3].params['np.log(1 + tau_lag)']/eres3[3].params['np.log(exports_lag)']))


print('\twriting output to LaTeX')

file = open(outpath + 'model_ecm.tex','w')

# header
#file.write('\\begin{landscape}\n')
file.write('\\begin{table}[p]\n')
file.write('\\footnotesize\n')
file.write('\\renewcommand{\\arraystretch}{1.2}\n')
file.write('\\begin{center}\n')
file.write("\\caption{Short and long-run trade elasticities in simulated data}\n")
file.write('\\label{tab:model_ecm}\n')
file.write('\\begin{tabular}{lcccc')
file.write('}')
file.write('\\toprule\n')
    
#colname1 = lambda s: '\\multicolumn{1}{b{1.2cm}}{\centering '+s+'}'
#colname2 = lambda s: '\\multicolumn{1}{b{1.3cm}}{\centering '+s+'}'
#colname3 = lambda s: '\\multicolumn{1}{b{1.5cm}}{\centering '+s+'}'
colname = lambda s: '&\\multicolumn{1}{b{2cm}}{\centering '+s+'}'

# column names
file.write('Dep. var.')
file.write(colname('No TPU (1980 anticipated)'))
file.write(colname('No TPU (1980 surprise)'))
file.write(colname('No TPU (from 1979 SS)'))
file.write(colname('Benchmark TPU model'))
#file.write('& \\multicolumn{1}{c}{No TPU (1980 anticipated)} & \\multicolumn{3}{1}{No TPU (1980 surprise)}& \\multicolumn{1}{c}{No TPU (from 1979 SS)}&\\multicolumn{1}{c}{With TPU}\\\\\n')
#file.write('\\cmidrule(rl){2-4}\\cmidrule(rl){5-7}\\cmidrule(rl){8-10}\\cmidrule(rl){11-13}\n')

    
#for i in range(3):
#    file.write('&'+colname1(r'Cross-\\section'))
#    file.write('&'+colname2(r'ECM\\restricted'))
#    file.write('&'+colname3(r'ECM\\unrestricted'))
file.write('\\\\\n\\midrule\n')

# numbers
#file.write('$\\tau_{gt}$')
#for i in range(4):
#    file.write('&%0.2f'%eres1b[i].params['np.log(1 + tau_applied)'])
#    file.write('&%0.2f&'%eres2[i].params['np.log(1 + tau_applied)'])
#file.write('\\\\\n')

file.write('$\\Delta\\tau_{gt}$')
for i in range(4):
    file.write('&%0.2f'%eres3[i].params['delta_tau'])
file.write('\\\\\n')
        
file.write('$v_{g,t-1}$')
for i in range(4):
    #file.write('&&%0.2f'%eres2[i].params['np.log(exports_lag)'])
    file.write('&%0.2f'%eres3[i].params['np.log(exports_lag)'])
file.write('\\\\\n')
        
file.write('$\\tau_{g,t-1}$')
for i in range(4):
    file.write('&%0.2f'%eres3[i].params['np.log(1 + tau_lag)'])
file.write('\\\\\n')

file.write('\\midrule\n')
file.write('LR')
for i in range(4):
    #file.write('&%0.2f'%(eres2[i].params['np.log(1 + tau_applied)']/(1-eres2[i].params['np.log(exports_lag)'])))
    file.write('&%0.2f'%(-eres3[i].params['np.log(1 + tau_lag)']/eres3[i].params['np.log(exports_lag)']))
file.write('\\\\\n')

file.write('LR/SR')
for i in range(4):
    #lr = eres2[i].params['np.log(1 + tau_applied)']/(1-eres2[i].params['np.log(exports_lag)'])
    #sr = eres2[i].params['np.log(1 + tau_applied)']
    #file.write('&%0.2f'%(lr/sr))
        
    lr = -eres3[i].params['np.log(1 + tau_lag)']/eres3[i].params['np.log(exports_lag)']
    sr = eres3[i].params['delta_tau']
    file.write('&%0.2f'%(lr/sr))
file.write('\\\\\n')


# footer
file.write('\\bottomrule\n')
file.write('\\end{tabular}\n')
file.write('\\end{center}\n')
file.write('\\normalsize\n')
file.write('\\end{table}\n')

file.close()
    

