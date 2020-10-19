# -*- coding: utf-8 -*-
"""
Created on Fri Apr 24 12:24:05 2020

@author: Abhimanyu Trakroo
"""

# Importing the relevant packages
# Importing the relevant packages
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.graphics.tsaplots as sgt
from statsmodels.tsa.arima_model import ARMA
from scipy.stats.distributions import chi2
import statsmodels.tsa.stattools as sts 
import seaborn as sns
sns.set()

## Importing the Data and Pre-processing 
## Importing the Data and Pre-processing 

raw_csv_data = pd.read_csv("............/Index2018.csv") 
df_comp=raw_csv_data.copy()
df_comp.date = pd.to_datetime(df_comp.date, dayfirst = True)
df_comp.set_index("date", inplace=True)
df_comp=df_comp.asfreq('b')
df_comp=df_comp.fillna(method='ffill')

df_comp['market_value']=df_comp.ftse

del df_comp['spx']
del df_comp['dax']
del df_comp['ftse']
del df_comp['nikkei']
size = int(len(df_comp)*0.8)
df, df_test = df_comp.iloc[:size], df_comp.iloc[size:]



## LLR Test
## LLR Test
## LLR Test

def LLR_test(mod_1, mod_2, DF=1):
    L1 = mod_1.fit().llf
    L2 = mod_2.fit().llf
    LR = (2*(L2-L1))
    p = chi2.sf(LR, DF).round(3)
    return p

## The DF-Test
## The DF-Test
## The DF-Test    

sts.adfuller(df.market_value)

## Using Returns

df['returns'] = df.market_value.pct_change(1).mul(100)
df = df.iloc[1:]

sts.adfuller(df.returns)

## ACF and PACF for Returns
## ACF and PACF for Returns

sgt.plot_acf(df.returns, lags=40, zero = False)
plt.title("ACF FTSE Returns", size=24)
plt.show()

sgt.plot_pacf(df.returns, lags = 40, zero = False, method = ('ols'))
plt.title("PACF FTSE Returns", size=24)
plt.show()

## AR(1) for Returns
## AR(1) for Returns

model_ret_ar_1 = ARMA(df.returns, order = (1,0))

results_ret_ar_1 = model_ret_ar_1.fit()

results_ret_ar_1.summary()

## Higher-Lag AR Models for Returns
## Higher-Lag AR Models for Returns

model_ret_ar_2 = ARMA(df.returns, order = (2,0))
results_ret_ar_2 = model_ret_ar_2.fit()
results_ret_ar_2.summary()

LLR_test(model_ret_ar_1, model_ret_ar_2)

model_ret_ar_3 = ARMA(df.returns, order = (3,0))
results_ret_ar_3 = model_ret_ar_3.fit()
results_ret_ar_3.summary()

LLR_test(model_ret_ar_2, model_ret_ar_3)

model_ret_ar_4 = ARMA(df.returns, order = (4,0))
results_ret_ar_4 = model_ret_ar_4.fit()
print(results_ret_ar_4.summary())
print ("LLR test: " + str(LLR_test(model_ret_ar_3, model_ret_ar_4)))

model_ret_ar_5 = ARMA(df.returns, order = (5,0))
results_ret_ar_5 = model_ret_ar_5.fit()
print(results_ret_ar_5.summary())
print ("LLR test: " + str(LLR_test(model_ret_ar_4, model_ret_ar_5)))


model_ret_ar_6 = ARMA(df.returns, order = (6,0))
results_ret_ar_6 = model_ret_ar_6.fit()
print(results_ret_ar_6.summary())
print ("LLR test: " + str(LLR_test(model_ret_ar_5, model_ret_ar_6)))


model_ret_ar_7 = ARMA(df.returns, order = (7,0))
results_ret_ar_7 = model_ret_ar_7.fit()
results_ret_ar_7.summary()


print (LLR_test(model_ret_ar_6, model_ret_ar_7))


## Normalizing Values
## Normalizing Values
## Normalizing Values

benchmark = df.market_value.iloc[0]


df['norm'] = df.market_value.div(benchmark).mul(100)

sts.adfuller(df.norm)

bench_ret = df.returns.iloc[0]
df['norm_ret'] = df.returns.div(bench_ret).mul(100)
sts.adfuller(df.norm_ret)

## Normalized Returns
## Normalized Returns
## Normalized Returns

model_norm_ret_ar_1 = ARMA (df.norm_ret, order=(1,0))
results_norm_ret_ar_1 = model_norm_ret_ar_1.fit()
results_norm_ret_ar_1.summary()

model_norm_ret_ar_2 = ARMA(df.norm_ret, order=(2,0))
results_norm_ret_ar_2 = model_norm_ret_ar_2.fit()
results_norm_ret_ar_2.summary()


model_norm_ret_ar_7 = ARMA(df.norm_ret, order=(7,0))
results_norm_ret_ar_7 = model_norm_ret_ar_7.fit()
results_norm_ret_ar_7.summary()

## Analysing the Residuals
## Analysing the Residuals
## Analysing the Residuals

df['res_ret'] = results_ret_ar_6.resid

df.res_ret.mean()

df.res_ret.var()

sts.adfuller(df.res_ret)

sgt.plot_acf(df.res_ret, zero = False, lags = 40)
plt.title("ACF Of Residuals for Returns",size=24)
plt.show()

df.res_ret.plot(figsize=(20,5))
plt.title("Residuals of Returns", size=24)
plt.show()
