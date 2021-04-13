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

##################################################################

df = pd.read_csv('../model/output/simul_agg_no_tpu.csv')

df['lv'] = np.log(df.exports)
df['ltau'] = np.log(df.tau_applied)
df['l_lv'] = df.groupby(['s','i'])['lv'].transform(lambda x:x.shift())
df['l_ltau'] = df.groupby(['s','i'])['ltau'].transform(lambda x:x.shift())
df['d_lv'] = df.lv - df.l_lv
df['d_ltau'] = df.ltau - df.l_ltau

df=df[(df.d_lv.notna())&(df.d_lv>-9999999)&(df.d_lv<9999999)]

formula = 'd_lv ~ l_lv + l_ltau + d_ltau'
print(smf.ols(formula=formula,data=df).fit(cov_type='HC0').summary())

