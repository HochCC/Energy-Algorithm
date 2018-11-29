import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.stats import norm
from sklearn.preprocessing import StandardScaler
from scipy import stats 
from sklearn.preprocessing import RobustScaler

df_train = pd.read_csv('../dataset/feature/energy_train_1029_features.csv')  
#df_train = df_train[df_train['vehicle_id'].isin([4])].copy()
 
df_train_sigle = pd.read_csv('../dataset/feature/train_feature/car12_features.csv', dtype={'phase': str, 'charge_mode': str})
df_train15 = pd.read_csv('../dataset/feature/train_feature/car15_features.csv', dtype={'phase': str, 'charge_mode': str})
df_train16 = pd.read_csv('../dataset/feature/train_feature/car16_features.csv', dtype={'phase': str, 'charge_mode': str})

sns.distplot(df_train['charge_energy'])
plt.figure()
plt.subplot(1,2,1)
sns.distplot(df_train15['charge_min_temp'])
plt.subplot(1,2,2)
sns.distplot(df_train16['charge_min_temp'])
plt.figure()
soc = df_train['charge_min_temp'].copy()
soc.dropna(axis=0, inplace=True)
sns.distplot(soc)

print("Skewness: %f" % df_train['charge_energy'].skew())
print("Kurtosis: %f" % df_train['charge_energy'].kurt()) 

var = 'dsoc' 
data = pd.concat([df_train['charge_energy'], df_train[var]], axis=1)
data.plot.scatter(x=var, y='charge_energy');

var = 'charge_hour'
data = pd.concat([df_train['charge_energy'], df_train[var]], axis=1)
data.plot.scatter(x=var, y='charge_energy');

var = 'vehicle_id'
data = pd.concat([df_train[var], df_train['dsoc']], axis=1)
f, ax = plt.subplots(figsize=(8, 6))
fig = sns.boxplot(x=var, y='dsoc', data=data)

var = 'vehicle_id'
data = pd.concat([df_train[var], df_train['charge_energy']], axis=1)
f, ax = plt.subplots(figsize=(8, 6))
fig = sns.boxplot(x=var, y='charge_energy', data=data)

var = 'vehicle_id'
data = pd.concat([df_train[var], df_train['charge_min_temp']], axis=1)
f, ax = plt.subplots(figsize=(8, 6))
fig = sns.boxplot(x=var, y='charge_min_temp', data=data)

var = 'vehicle_id'
data = pd.concat([df_train[var], df_train['charge_max_temp']], axis=1)
f, ax = plt.subplots(figsize=(8, 6))
fig = sns.boxplot(x=var, y='charge_max_temp', data=data)

var = 'vehicle_id'
data = pd.concat([df_train[var], df_train['charge_end_I']], axis=1)
f, ax = plt.subplots(figsize=(8, 6))
fig = sns.boxplot(x=var, y='charge_end_I', data=data)

var = 'vehicle_id'
data = pd.concat([df_train[var], df_train['charge_end_U']], axis=1)
f, ax = plt.subplots(figsize=(8, 6))
fig = sns.boxplot(x=var, y='charge_end_U', data=data)

var = 'vehicle_id'
data = pd.concat([df_train[var], df_train['charge_end_soc']], axis=1)
f, ax = plt.subplots(figsize=(8, 6))
fig = sns.boxplot(x=var, y='charge_end_soc', data=data)

var = 'vehicle_id'
data = pd.concat([df_train[var], df_train['charge_start_soc']], axis=1)
f, ax = plt.subplots(figsize=(8, 6))
fig = sns.boxplot(x=var, y='charge_start_soc', data=data)

var = 'vehicle_id'
data = pd.concat([df_train[var], df_train['dmileage']], axis=1)
f, ax = plt.subplots(figsize=(8, 6))
fig = sns.boxplot(x=var, y='dmileage', data=data)
 
corrmat = df_train.corr()
f, ax = plt.subplots(figsize=(12, 9))
sns.heatmap(corrmat, vmax=.8, square=True)

# car10
df_train_12 = pd.read_csv('../dataset/feature/train_feature/car12_features.csv') 
chargemode_dummies = pd.get_dummies(df_train_12['charge_mode'], prefix='mode', prefix_sep='_')
hour_dummies = pd.get_dummies(df_train_12['hour'], prefix='hour', prefix_sep='_')
week_dummies = pd.get_dummies(df_train_12['week'], prefix='week', prefix_sep='_') 
month_dummies = pd.get_dummies(df_train_12['month'], prefix='month', prefix_sep='_') 
phase_dummies = pd.get_dummies(df_train_12['phase'], prefix='phase', prefix_sep='_')
df_train_12_dummi = pd.concat([df_train_12, chargemode_dummies, hour_dummies, week_dummies, month_dummies,phase_dummies], axis=1)
df_train_12_dummi.drop(['charge_mode', 'hour', 'week', 'month', 'phase'], axis=1, inplace=True)
 
f,ax = plt.subplots(figsize=(18, 18))
sns.heatmap(df_train_12_dummi.corr(), annot=False, linewidths=.2, fmt= '.1f',ax=ax)

plt.figure(figsize=(25, 25))
sns.set(style="white")
corr = df_train_12_dummi.corr()  
mask = np.zeros_like(corr, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True 
cmap = sns.diverging_palette(220, 10, as_cmap=True) 
sns.heatmap(corr, mask=mask, cmap=cmap, center=0, square=True, linewidths=0.2)
  
sns.set()
df_train_sigle.dropna(axis=0, how='any', inplace=True)
df_train_sigle['P/kW'] = df_train_sigle['charge_energy'] / df_train_sigle['charge_hour']
cols = ['charge_energy', 'dsoc', 'charge_hour', 'dsoc/hour', 'P/kW', 'dU', 'dU/dsoc', 'phase']
sns.pairplot(df_train_sigle[cols], diag_kind='kde', hue='phase') 
plt.show()

total = df_train.isnull().sum().sort_values(ascending=False)
percent = (df_train.isnull().sum()/df_train.isnull().count()).sort_values(ascending=False)
missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
print(missing_data)

#standardizing data
saleprice_scaled = RobustScaler().fit_transform(df_train['charge_energy'][:,np.newaxis]);
low_range = saleprice_scaled[saleprice_scaled[:,0].argsort()][:10]
high_range= saleprice_scaled[saleprice_scaled[:,0].argsort()][-10:]
print('outer range (low) of the distribution:')
print(low_range)
print('\nouter range (high) of the distribution:')
print(high_range)

#histogram and normal probability plot
fig = plt.figure() 
plt.subplot(1,2,1)
sns.distplot(df_train['charge_energy'], fit=norm);
plt.subplot(1,2,2)
res = stats.probplot(df_train['charge_energy'], plot=plt)

#applying log transformation
df_train['charge_energy'] = np.log1p(df_train['charge_energy'])
#transformed histogram and normal probability plot
fig = plt.figure()
plt.subplot(1,2,1)
sns.distplot(df_train['charge_energy'], fit=norm);
plt.subplot(1,2,2)
res = stats.probplot(df_train['charge_energy'], plot=plt)
 

#create column for new variable (one is enough because it's a binary categorical feature)
#if area>0 it gets 1, for area==0 it gets 0
fig = plt.figure()
plt.subplot(1,2,1)
df_train['HasBsmt'] = pd.Series(len(df_train['charge_energy']), index=df_train.index)
df_train['HasBsmt'] = 0 
df_train.loc[df_train['charge_energy']>0,'HasBsmt'] = 1
#transform data
df_train.loc[df_train['HasBsmt']==1,'charge_energy'] = np.log(df_train['charge_energy'])
#histogram and normal probability plot
sns.distplot(df_train[df_train['charge_energy']>0]['charge_energy'], fit=norm);
plt.subplot(1,2,2)
res = stats.probplot(df_train[df_train['charge_energy']>0]['charge_energy'], plot=plt)  
    
train_targets = pd.DataFrame(df_train['charge_energy'], columns=['charge_energy'])
y_scaler = RobustScaler()  
#applying log transformation
df_train['charge_energy'] = y_scaler.fit_transform(np.array(train_targets)) 
#transformed histogram and normal probability plot
fig = plt.figure()
plt.subplot(1,2,1)
sns.distplot(df_train['charge_energy'], fit=norm);
plt.subplot(1,2,2)
res = stats.probplot(df_train['charge_energy'], plot=plt)

 
y_scaler = StandardScaler()  
#applying log transformation
df_train['charge_energy'] = y_scaler.fit_transform(np.array(train_targets)) 
#transformed histogram and normal probability plot
fig = plt.figure()
plt.subplot(1,2,1)
sns.distplot(df_train['charge_energy'], fit=norm);
plt.subplot(1,2,2)
res = stats.probplot(df_train['charge_energy'], plot=plt)