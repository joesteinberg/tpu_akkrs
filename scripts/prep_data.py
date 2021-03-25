#################################################################
# imports, etc.

import numpy as np
import pandas as pd
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
from statsmodels.api import OLS
from statsmodels.formula.api import ols
import patsy

mpl.rc('text', usetex=True)
mpl.rc('savefig',bbox='tight')
mpl.rc('savefig',format='pdf')
mpl.rc('font',**{'family':'serif','serif':['Palatino'],'size':8})
mpl.rc('font',size=8)
mpl.rc('lines',linewidth=1)

inpath = '/home/joseph/Research/ongoing_projects/tpu_akkrs/us_imports_data/Stata Files/'

def fillin(df,cols):  
    tmp = pd.MultiIndex.from_product([df[c].unique().tolist() for c in cols], names = cols)
    tmp2 = pd.DataFrame(index = tmp).reset_index()
    df2 = pd.merge(left=df,right=tmp2,how='right',on=cols,indicator=True)
    df2['_fillin'] = df2._merge=='right_only'
    df2.drop('_merge',axis=1,inplace=True)
    return df2

def interp_nearest(x):
    if(x.count()>=2):
        return x.interpolate(method='nearest')
    else:
        return np.nan*x

#################################################################
# NNTR rates

nntr = pd.read_stata(inpath + 'spread_hs6.dta').drop('sitc2',axis=1)
nntr['year'] = nntr.year.astype(int)

hs96 = pd.read_stata(inpath + 'SITC2-HS96.dta')
hs92 = pd.read_stata(inpath + 'SITC2-HS92.dta')

nntr = pd.merge(left=nntr,right=hs96,how='left',on='hs6',indicator=True).rename(columns={'sitc2':'sitc2_orig'})
nntr = nntr.loc[nntr._merge != 'right only',:].reset_index(drop=True)
nntr.drop('_merge',axis=1,inplace=True)

nntr = pd.merge(left=nntr,right=hs92,how='left',on='hs6',indicator=True)
nntr = nntr.loc[nntr._merge != 'right only',:].reset_index(drop=True)
nntr['sitc2_orig'].fillna(nntr.sitc2)
nntr.drop(['sitc2','_merge'],axis=1,inplace=True)
nntr.rename(columns={'sitc2_orig':'sitc2'},inplace=True)

nntr['i'] = nntr['sitc2'].str[0:2]
nntr = nntr.loc[nntr.year==2001]
nntr = nntr.groupby(['i'])['nntr'].median().reset_index()

#############################################################################
# Applied tariffs

df = pd.read_stata(inpath + 'imports_sitc2_74-09.dta')[['cty','year','sitc2','tar_jst']]
df['year']=df.year.astype(int)
df['chn'] = df.cty=='CHINA'
df = fillin(df,['chn','sitc2','year'])
df = df[~( (df._fillin==1) & (df.chn==0) )].reset_index(drop=True)
df.loc[df.chn==1,'cty'] = 'CHINA'

comm = ['POLAND','GERMAN','YUGOSLAV','ALBANIA','BULGARIA','ROMANIA','HUNGARY','CUBA','VIETNAM','MONGOLIA']
df['ctycomm'] = df.cty.isin(comm)
df['temp_pre80'] = df.year<1980

# fill in missing data for bilateral tariffs
df['temp1'] = df.groupby(['chn','sitc2','temp_pre80'])['tar_jst'].transform(lambda x:interp_nearest(x))
df['temp_tar'] = df.tar_jst
df.loc[(df.ctycomm==0),'temp_tar'] = None
df['temp2'] = df.groupby(['sitc2','temp_pre80'])['temp_tar'].transform(lambda x: x.mean())
df['temp3'] = df.groupby(['chn','year'])['tar_jst'].transform(lambda x:x.mean())
df['tarfull_jst'] = df.tar_jst
df.loc[(df.chn==1) & (df.tarfull_jst.isna()),'tarfull_jst'] = df.loc[(df.chn==1) & (df.tarfull_jst.isna()),'temp1']
df.loc[(df.chn==1) & (df.tarfull_jst.isna()),'tarfull_jst'] = df.loc[(df.chn==1) & (df.tarfull_jst.isna()),'temp2']
df.loc[(df.chn==1) & (df.tarfull_jst.isna()),'tarfull_jst'] = df.loc[(df.chn==1) & (df.tarfull_jst.isna()),'temp3']

############################################################################
# now drill down and export the Chinese data for use in C program

df = df[df['cty']=='CHINA'].reset_index(drop=True)
df['i'] = df['sitc2'].str[0:2]
df = df[df.i.str[0] != '0'].reset_index(drop=True)
df = df[df.i.str[0] != '1'].reset_index(drop=True)
df = df[df.i.str[0] != '2'].reset_index(drop=True)
df = df[df.i.str[0] != '3'].reset_index(drop=True)
df = df[df.i.str[0] != '4'].reset_index(drop=True)
df = df[df.i.str[0] != '9'].reset_index(drop=True)
df = df[df.i != '56'].reset_index(drop=True)
df = df[df.i != '60'].reset_index(drop=True)
df = df[df.i != '80'].reset_index(drop=True)

dfi = df.groupby(['i','year'])[['tarfull_jst']].mean().reset_index()
dfi = fillin(dfi,['i','year'])

dfi.sort_values(by=['i','year'],ascending=[True,True],inplace=True)
dfi.reset_index(inplace=True,drop=True)

dfi['temp_pre80'] = dfi.year<1980
dfi['tar_temp1'] = dfi.groupby(['i','temp_pre80'])['tarfull_jst'].transform(lambda x:x.interpolate('nearest',limit=2))
dfi['tar_temp2'] = dfi.groupby(['i','temp_pre80'])['tarfull_jst'].transform(lambda x:x.mean())

dfi['tar_final'] = 0

dfi.loc[dfi.temp_pre80==0,'tar_final'] = dfi.loc[dfi.temp_pre80==0,'tarfull_jst']
dfi.loc[(dfi.temp_pre80==0)&(dfi.tar_final.isna()),'tar_final'] = dfi.loc[(dfi.temp_pre80==0)&(dfi.tar_final.isna()),'tar_temp1']

dfi.loc[dfi.temp_pre80==1,'tar_final'] = dfi.loc[dfi.temp_pre80==1,'tar_temp2']


#dfi[c].fillna(dfi[c+'temp1'],inplace=True)

#dfi[c+'temp2'] = dfi.groupby(['i','temp_pre80'])[c].transform(lambda x:x.interpolate('bfill'))
#dfi[c].fillna(dfi[c+'temp2'],inplace=True)

#dfi[c+'temp3'] = dfi.groupby(['i','temp_pre80'])[c].transform(lambda x:x.interpolate('ffill'))
#dfi[c].fillna(dfi[c+'temp3'],inplace=True)


dfi = pd.merge(left=dfi,right=nntr,how='left',on='i')
dfi = dfi[dfi.nntr.notna()].reset_index(drop=True)
    
dfi['icat'] = pd.Categorical(dfi['i'])
dfi['inum'] = dfi.icat.cat.codes
dfi.drop('icat',axis=1)

dfi['year'] = dfi.year.astype(int) - 1971
dfi[['inum','i','year','tar_final','nntr']].to_csv('tariff_data.csv',sep=' ',header=False, index=False)

print(dfi.groupby('year')['tar_final'].mean())


#############################################################################
# Applied tariffs 2d

df2 = pd.read_stata(inpath + 'imports_2dsitc2_74-09.dta')[['cty','year','sitc2_2d','tar_jst']]
df2['year']=df2.year.astype(int)
df2['chn'] = df2.cty=='CHINA'
df2 = fillin(df2,['chn','sitc2_2d','year'])
df2 = df2[~( (df2._fillin==1) & (df2.chn==0) )].reset_index(drop=True)
df2.loc[df2.chn==1,'cty'] = 'CHINA'

comm = ['POLAND','GERMAN','YUGOSLAV','ALBANIA','BULGARIA','ROMANIA','HUNGARY','CUBA','VIETNAM','MONGOLIA']
df2['ctycomm'] = df2.cty.isin(comm)
df2['temp_pre80'] = df2.year<1980

# fill in missing data for bilateral tariffs
df2['temp1'] = df2.groupby(['chn','sitc2_2d','temp_pre80'])['tar_jst'].transform(lambda x:interp_nearest(x))
df2['temp_tar'] = df2.tar_jst
df2.loc[(df2.ctycomm==0),'temp_tar'] = None
df2['temp2'] = df2.groupby(['sitc2_2d','temp_pre80'])['temp_tar'].transform(lambda x: x.mean())
df2['temp3'] = df2.groupby(['chn','year'])['tar_jst'].transform(lambda x:x.mean())
df2['tarfull_jst'] = df2.tar_jst
df2.loc[(df2.chn==1) & (df2.tarfull_jst.isna()),'tarfull_jst'] = df2.loc[(df2.chn==1) & (df2.tarfull_jst.isna()),'temp1']
df2.loc[(df2.chn==1) & (df2.tarfull_jst.isna()),'tarfull_jst'] = df2.loc[(df2.chn==1) & (df2.tarfull_jst.isna()),'temp2']
df2.loc[(df2.chn==1) & (df2.tarfull_jst.isna()),'tarfull_jst'] = df2.loc[(df2.chn==1) & (df2.tarfull_jst.isna()),'temp3']

############################################################################
# now drill down and export the Chinese data for use in C program

df2 = df2[df2['cty']=='CHINA'].reset_index(drop=True)
df2['i'] = df2['sitc2_2d']
df2 = df2[df2.i.str[0] != '0'].reset_index(drop=True)
df2 = df2[df2.i.str[0] != '1'].reset_index(drop=True)
df2 = df2[df2.i.str[0] != '2'].reset_index(drop=True)
df2 = df2[df2.i.str[0] != '3'].reset_index(drop=True)
df2 = df2[df2.i.str[0] != '4'].reset_index(drop=True)
df2 = df2[df2.i.str[0] != '9'].reset_index(drop=True)
df2 = df2[df2.i != '56'].reset_index(drop=True)
df2 = df2[df2.i != '60'].reset_index(drop=True)
df2 = df2[df2.i != '80'].reset_index(drop=True)

dfi = df2[['i','year','tarfull_jst']]
dfi = fillin(dfi,['i','year'])

dfi.sort_values(by=['i','year'],ascending=[True,True],inplace=True)
dfi.reset_index(inplace=True,drop=True)

dfi['temp_pre80'] = dfi.year<1980
dfi['tar_temp1'] = dfi.groupby(['i','temp_pre80'])['tarfull_jst'].transform(lambda x:x.interpolate('nearest',limit=2))
dfi['tar_temp2'] = dfi.groupby(['i','temp_pre80'])['tarfull_jst'].transform(lambda x:x.mean())

dfi['tar_final'] = 0

dfi.loc[dfi.temp_pre80==0,'tar_final'] = dfi.loc[dfi.temp_pre80==0,'tarfull_jst']
dfi.loc[(dfi.temp_pre80==0)&(dfi.tar_final.isna()),'tar_final'] = dfi.loc[(dfi.temp_pre80==0)&(dfi.tar_final.isna()),'tar_temp1']

dfi.loc[dfi.temp_pre80==1,'tar_final'] = dfi.loc[dfi.temp_pre80==1,'tar_temp2']


#dfi[c].fillna(dfi[c+'temp1'],inplace=True)

#dfi[c+'temp2'] = dfi.groupby(['i','temp_pre80'])[c].transform(lambda x:x.interpolate('bfill'))
#dfi[c].fillna(dfi[c+'temp2'],inplace=True)

#dfi[c+'temp3'] = dfi.groupby(['i','temp_pre80'])[c].transform(lambda x:x.interpolate('ffill'))
#dfi[c].fillna(dfi[c+'temp3'],inplace=True)


dfi = pd.merge(left=dfi,right=nntr,how='left',on='i')
dfi = dfi[dfi.nntr.notna()].reset_index(drop=True)
    
dfi['icat'] = pd.Categorical(dfi['i'])
dfi['inum'] = dfi.icat.cat.codes
dfi.drop('icat',axis=1)

dfi['year'] = dfi.year.astype(int) - 1971
dfi[['inum','i','year','tar_final','nntr']].to_csv('tariff_data2.csv',sep=' ',header=False, index=False)

print(dfi.groupby('year')['tar_final'].mean())
