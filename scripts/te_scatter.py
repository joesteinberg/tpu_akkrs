#################################################################
# imports, etc.

import sys
import numpy as np
import pandas as pd
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import numpy.polynomial.polynomial as poly

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
    elif(reform_flag==5):
        fname = 'simul_agg_det5_baseline.csv' 
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
df = [load_data(0),load_data(1),load_data(4),load_data(3),load_data(2),load_data(5)]
df[4].tau_applied = df[4].tau_nntr

#######################

fig,ax = plt.subplots(1,1,figsize=(4,4))

tmp = df[4].loc[df[4].y==df[4].y.max(),:]
ax.scatter(np.log(1 + tmp.tau_nntr),np.log(tmp.exports),label='NNTR steady state',s=3,alpha=0.25,color=colors[1])

tmp = df[3].loc[df[3].y==df[3].y.max(),:]
ax.scatter(np.log(1 + tmp.tau_applied),np.log(tmp.exports),label='Baseline steady state',s=3,alpha=0.25,color=colors[2])

tmp = df[5].loc[df[5].y==df[5].y.max(),:]
ax.scatter(np.log(1 + tmp.tau_applied),np.log(tmp.exports),label='MFN steady state',s=3,alpha=0.25,color=colors[0])

tmp = df[4].loc[df[4].y==df[4].y.max(),:]
tmp.sort_values(by='tau_applied',inplace=True)
x = np.log(1 + tmp.tau_applied)
p = poly.polyfit(x,np.log(tmp.exports),4)
y = poly.polyval(x, p)
ax.plot(x,y,color='black',linestyle='--',label='best polynomial fit')
        

ax.set_xlabel(r'$\log \tau$')
ax.set_ylabel(r'$\log exports$')
ax.legend(loc='lower right',prop={'size':6})
fig.subplots_adjust(hspace=0.2,wspace=0.25)
plt.savefig('model_fig_scatter_elast_cross_sec.pdf',bbox_inches='tight')

plt.close('all')



fig,ax = plt.subplots(1,1,figsize=(4,4))
d = poly.polyder(p)
y = poly.polyval(x,d)

ax.plot(x,y,color='black',linestyle='-',label='derivative')
ax.set_xlabel(r'$d \log \tau$')
ax.set_ylabel(r'$d \log exports$')
#ax.legend(loc='lower right',prop={'size':6})
fig.subplots_adjust(hspace=0.2,wspace=0.25)
plt.savefig('model_fig_elast_cross_sec_deriv.pdf',bbox_inches='tight')

df_fig = pd.DataFrame({'d_log_tau':x,'d_log_exports':y})
df_fig.to_csv(outpath+'model_fig_scatter_elast_cross_sec.csv',index=False)


plt.close('all')





