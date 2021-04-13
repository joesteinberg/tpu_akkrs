import numpy as np
import pandas as pd

df = pd.read_stata('../us_imports_data/Stata Files/path_simulation_sitc.dta')
df['year'] = df.year.astype(int)
df=df[df.year<2009]

def fillin(df,cols):  
    tmp = pd.MultiIndex.from_product([df[c].unique().tolist() for c in cols], names = cols)
    tmp2 = pd.DataFrame(index = tmp).reset_index()
    df2 = pd.merge(left=df,right=tmp2,how='right',on=cols,indicator=True)
    df2['_fillin'] = df2._merge=='right_only'
    df2.drop('_merge',axis=1,inplace=True)
    return df2

df = fillin(df,['sitc','year']).drop('_fillin',axis=1)

#df09 = df[['sitc']].drop_duplicates()
#df09['year'] = 2009
#df=df.append(df09)

df.sort_values(by=['sitc','year'],ascending=[True,True],inplace=True)
df.reset_index(drop=True,inplace=True)

for c in ['mfn','nntr','gap']:
    df[c] = df.groupby('sitc')[c].transform(lambda x: x.fillna(method='bfill'))
    df[c] = df.groupby('sitc')[c].transform(lambda x: x.fillna(method='ffill'))
    df[c] = df.groupby('sitc')[c].transform(lambda x: x.interpolate(method='nearest'))

gap_tmp = df.loc[df.year==2001,['sitc','year','mfn','nntr']]
gap_tmp['gap_tmp'] = gap_tmp.nntr-gap_tmp.mfn
df=pd.merge(left=df,right=gap_tmp[['sitc','gap_tmp']],how='left',on=['sitc'])
df.loc[df.gap.isna(),'gap'] = df.loc[df.gap.isna(),'gap_tmp']
df.drop('gap_tmp',axis=1)



    #df[c+'_tmp'] = df.groupby('sitc')[c].apply(lambda x: x.interpolate(method='bfill'))
    #df[c].fillna(df[c+'_tmp'],inplace=True)
    #df.drop(c+'_tmp',axis=1)

    #df[c+'_tmp'] = df.groupby('sitc')[c].apply(lambda x: x.interpolate(method='ffil'))
    #df[c].fillna(df[c+'_tmp'],inplace=True)
    #df.drop(c+'_tmp',axis=1)


df['icat'] = pd.Categorical(df['sitc'])
df['inum'] = df.icat.cat.codes
df.drop('icat',axis=1)

df[['inum','sitc','year','mfn','nntr','gap']].to_csv('path_simulation.csv',sep=' ',header=False, index=False)




