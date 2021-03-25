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

#################################################################
# imports, etc.

inpath = '/home/joseph/Research/ongoing_projects/tpu_akkrs/us_imports_data/Stata Files/'
df = pd.read_stata(inpath + 'imports_sitc2_74-09.dta')
df['year'] = df.year.astype(int)

#################################################################
# repeat KK steps

def tag_(x):
    y=0*x
    y.iloc[0]=1
    return y

def tag(df,cols,name):
    df[name] = 0
    df[name] = df.groupby(cols)[name].transform(lambda x: tag_(x))

def fillin(df,cols,emptyval):  
    tmp = pd.MultiIndex.from_product([df[c].unique().tolist() for c in cols], names = cols)
    tmp2 = pd.DataFrame(index = tmp).reset_index()
    df2 = pd.merge(left=df,right=tmp2,how='right',on=cols,indicator=True)
    df2 = df2.fillna(emptyval)
    df2['_fillin'] = df2._merge=='right_only'
    df2.drop('_merge',axis=1,inplace=True)
    return df2

def interp_nearest(x):
    if(x.count()>=2):
        return x.interpolate(method='nearest')
    else:
        return np.nan*x

def wavg(x,wgts):
    if wgts.sum()>0.0:
        return np.average(x,wgts)
    else:
        return np.average(x)

df['exsoviet'] = 0
exsoviet = ['BELARUS','UZKBEKIST','UKRAINE','TURKMENI','TAJIKIST','MOLDOVA','LITHUANI','LATVIA','KYRGYZST','KAZAKHST','ESTONIA','ARMENIA','GEORGIA','AZERBAIJ']
df.loc[df.cty.isin(exsoviet),'exsoviet'] = 1

title4 = ['CZECHREP','SLOVAKIA','ROMANIA','HUNGARY']
df['title4'] = 0
df.loc[(df.exsoviet==1) | (df.cty.isin(title4)),'title4'] = 1

df.loc[df.cty=='GERMAN_E','cty'] = 'GERMAN'

df=df[df.year!=2009].reset_index(drop=True)

df['v_st'] = df.groupby(['sitc2','year'])['v_jst'].transform(lambda x:x.sum())
df['v_jt'] = df.groupby(['cty','year'])['v_jst'].transform(lambda x:x.sum())
df['v_t'] = df.groupby('year')['v_jst'].transform(lambda x:x.sum())

df['vjsh_jst'] = df.v_jst/df.v_st
df['vjsh_jt'] = df.v_jt/df.v_t
df['vssh_jt'] = df.v_st/df.v_t

df['duties_jt'] = df.groupby(['cty','year'])['duties_jst'].transform(lambda x:x.sum())
df['duties_st'] = df.groupby(['sitc2','year'])['duties_jst'].transform(lambda x:x.sum())

df['tar_jt'] = df.duties_jt/df.v_jt
df['tar_st'] = df.duties_st/df.v_st
df.loc[df.v_jt<1.0e-8,'tar_jt'] = 0
df.loc[df.v_st<1.0e-8,'tar_st'] = 0

df['tar_unwgt_st'] = df.groupby(['sitc2','year'])['tar_jst'].transform(lambda x:x.mean())
df['tar_unwgt_jt'] = df.groupby(['cty','year'])['tar_jst'].transform(lambda x:x.mean())
df['tar_sd_jt'] = df.groupby(['cty','year'])['tar_jst'].transform(lambda x:x.std())
df['tar_cov_jt'] = df.tar_unwgt_jt/df.tar_sd_jt
df['tar_sdp_jt'] = df.tar_unwgt_jt + df.tar_sd_jt
df['tar_sdm_jt'] = df.tar_unwgt_jt - df.tar_sd_jt

df.rename(columns={'nntr_med':'nntrmed01_s','s_med':'tarspd01_s'},inplace=True)
df['tarspd_jst'] = df.nntrmed01_s - df.tar_jst
df['tarspd_st'] = df.nntrmed01_s - df.tar_st

#df.columns = df.columns.str.replace("unwgt", "unwgt_jst")

for c in df.columns:
    if c[0]=='v':
        df['l'+c] = np.log(df[c])

    if c[0:3]=='tar':
        df['l'+c] = np.log(1+df[c])

df['post'] = df.year>2000

tag(df,['cty','year'],'tag_jt')
tag(df,['sitc2','year'],'tag_st')
tag(df,['cty'],'tag_j')
tag(df,['sitc2'],'tag_s')
tag(df,['year'],'tag_t')

df['chn'] = df.cty=='CHINA'
df['sitc2_1d'] = df.sitc2.str[0:1]
df['sitc2_2d'] = df.sitc2.str[0:2]
df['sitc2_3d'] = df.sitc2.str[0:3]

df['temp1'] = df.groupby(['sitc2','chn','post'])['year'].transform(lambda x:x.count())
df.loc[~((df.chn==1)&(df.post==0)),'temp1']=None

df['temp2'] = df.groupby('sitc2')['temp1'].transform(lambda x: x.mean())
df['sample_balancedchn_s'] = df.temp2>=6
df.drop(['temp1','temp2'],axis=1,inplace=True)

df['temp1'] = df.groupby(['sample_balancedchn_s','year'])['v_jst'].transform(lambda x:x.sum())
df['v_balancedchnsh_st'] = df.temp1/df.v_t
df.drop('temp1',axis=1,inplace=True)

tag(df,['sitc2','chn','year'],'temptag')
df.loc[~((df.chn==1)&(df.year<1980)),'temptag'] = None
df['temp2'] = df.groupby('sitc2')['temptag'].transform(lambda x:x.sum())
df['sample_pre1980chn_s'] = df.temp2==6
df.drop(['temp2','temptag'],axis=1,inplace=True)

df['temp1'] = df.groupby(['sample_pre1980chn_s','year'])['v_jst'].transform(lambda x:x.sum())
df['v_pre1980chn_s'] = df.temp1/df.v_t
df.drop('temp1',axis=1,inplace=True)


df = fillin(df,['chn','year','sitc2'],np.nan)
df = df[~( (df._fillin==1) & (df.chn==0) )].reset_index(drop=True)
df.loc[df.chn==1,'cty'] = 'CHINA'

comm = ['POLAND','GERMAN','YUGOSLAV','ALBANIA','BULGARIA','ROMANIA','HUNGARY','CUBA','VIETNAM','MONGOLIA']
df['ctycomm'] = df.cty.isin(comm)

df['temp_pre80'] = df.year<1980
df['temp1'] = df.groupby(['chn','sitc2','temp_pre80'])['tar_jst'].transform(lambda x:interp_nearest(x))
df['temp_tar'] = df.tar_jst
df.loc[(df.ctycomm==0),'temp_tar'] = None
df['temp2'] = df.groupby(['sitc2','temp_pre80'])['temp_tar'].transform(lambda x: x.mean())
df['temp3'] = df.groupby(['chn','year'])['tar_jst'].transform(lambda x:x.mean())

df['tarfull_jst'] = df.tar_jst
df.loc[(df.chn==1) & (df.tarfull_jst.isna()),'tarfull_jst'] = df.loc[(df.chn==1) & (df.tarfull_jst.isna()),'temp1']
df.loc[(df.chn==1) & (df.tarfull_jst.isna()),'tarfull_jst'] = df.loc[(df.chn==1) & (df.tarfull_jst.isna()),'temp2']
df.loc[(df.chn==1) & (df.tarfull_jst.isna()),'tarfull_jst'] = df.loc[(df.chn==1) & (df.tarfull_jst.isna()),'temp3']

df['temp_tar'] = df.tarfull_jst
df.loc[~((df.chn==1)&(df.year<1980)),'temp_tar']=None
df['temp6'] = df.groupby(['sitc2'])['temp_tar'].transform(lambda x:x.mean())

df['temp_tar'] = df.tarfull_jst
df.loc[~((df.chn==1)&(df.year>=1980)),'temp_tar'] = None
df['temp7'] = df.groupby(['sitc2'])['temp_tar'].transform(lambda x:x.mean())

df['sample_full'] = (pd.notnull(df.temp6)) & (pd.notnull(df.temp7))
df.loc[(df.chn==1)&(df.sample_full==0),'tarfull_jst'] = None

df['temp_year'] = df.year
df.loc[df.v_jst.isna(),'temp_year']=None

df['temp4'] = df.groupby(['cty','sitc2'])['temp_year'].transform(lambda x:x.max())
df['temp_v'] = df.v_jst
df.loc[~(df.year==df.temp4),'temp_v']=None
df['vfixmax_js'] = df.groupby(['cty','sitc2'])['temp_v'].transform(lambda x:x.mean())

df['temp5'] = df.groupby(['cty','sitc2'])['temp_year'].transform(lambda x:x.min())
df['temp_v'] = df.v_jst
df.loc[~(df.year==df.temp5),'temp_v']=None
df['vfixmin_js'] = df.groupby(['cty','sitc2'])['temp_v'].transform(lambda x:x.mean())

df.drop(['temp1','temp2','temp3','temp4','temp5','temp6','temp7','temp_year','temp_v','temp_tar','temp_pre80'],axis=1,inplace=True)

#df['tar_fixmax_jt'] = df.groupby(['cty','year'])['tarfull_jst'].transform(lambda x:wavg(x,df.loc[x.index,'vfixmax_js']))
#df['tar_fixmin_jt'] = df.groupby(['cty','year'])['tarfull_jst'].transform(lambda x:wavg(x,df.loc[x.index,'vfixmin_js']))
#df['tar_unwgtfull_jt'] = df.groupby(['cty','year'])['tarfull_jst'].transform(lambda x:np.mean())
#df['ltar_fixmax_jt'] = np.log(1+df.tar_fixmax_jt)
#df['ltar_fixmin_jt'] = np.log(1+df.tar_fixmin_jt)
#df['ltar_unwgtfull_jt'] = np.log(1+df.tar_unwgtfull_jt)

df['tarfull_unwgt_jt'] = df.groupby(['cty','year'])['tarfull_jst'].transform(lambda x:x.mean())
df['tarfull_sd_jt'] = df.groupby(['cty','year'])['tarfull_jst'].transform(lambda x:x.std())
df['tarfull_cov_jt'] = df.tarfull_unwgt_jt/df.tarfull_sd_jt
df['tarfull_sdp_jt'] = df.tarfull_unwgt_jt + df.tarfull_sd_jt
df['tarfull_sdm_jt'] = df.tarfull_unwgt_jt - df.tarfull_sd_jt
df['tarfull_p25_jt'] = df.groupby(['cty','year'])['tarfull_jst'].transform(lambda x:x.quantile(0.25))
df['tarfull_p50_jt'] = df.groupby(['cty','year'])['tarfull_jst'].transform(lambda x:x.quantile(0.5))
df['tarfull_p75_jt'] = df.groupby(['cty','year'])['tarfull_jst'].transform(lambda x:x.quantile(0.75))


df['temp_lv'] = df['lvjsh_jt']
df.loc[~(df.year==1974),'temp_lv']=np.nan
df['temp1'] = df.groupby('cty')['temp_lv'].transform(lambda x:x.mean())
df['temp2'] = df.lvjsh_jt - df.temp1

tmp = df.loc[(df.cty=='CHINA')&(df.tag_jt==1),;].
tmp.sort_values(by='year',ascending=True,inplace=True)
fig,ax = plt.subplots(1,1,figsize=(4,4))
ax.plot(tmp.year,tmp.tar_unwgt_jt,label='Unweighted')
ax.plot(tmp.year,tmp.tarfull_p25_jt,label='p25')
ax.plot(tmp.year,tmp.tarfull_p50_jt,label='Median')
ax.plot(tmp.year,tmp.tarfull_p75_jt,label='p75')
ax2=ax.twinx()
ax2.plot(tmp.year,tmp.lvjsh_jt,label='Import share')
fig.subplots_adjust(hspace=0.2,wspace=0.25)
plt.savefig('importshare_vs_tariffs1.pdf',bbox_inches='tight')
plt.close('all')    
