#!/usr/bin/env python
# coding: utf-8

# In[61]:


#Import Libraries
import pandas as pd
import numpy as np
import seaborn as sns

import matplotlib.pyplot as plt
import matplotlib.dates as dt

get_ipython().run_line_magic('matplotlib', 'inline')

from scipy.stats import norm, skew #for some statistics
from scipy import stats #qqplot
import statsmodels.api as sm #for decomposing the trends, seasonality etc.

from matplotlib import rcParams

from statsmodels.tsa.statespace.sarimax import SARIMAX #the big daddy


# In[62]:


df  = pd.read_csv('SalesData.csv', parse_dates = {'date_col' : [ "MTH", "YR"]})
pd.set_option('display.float_format', lambda x: '%.3f' % x)


# In[63]:


df.head()


# In[64]:


df.dtypes


# In[65]:


#checking number of columns and rows
df.shape


# In[66]:


# Check any number of columns with NaN
print(df.isnull().any().sum(), ' / ', len(df.columns))
# Check any number of data points with NaN
print(df.isnull().any(axis=1).sum(), ' / ', len(df))


# In[67]:


#seems like we dont have any column or row with null value
# lets have a look at distribution plot of SALESAMOUNT
sns.distplot(df['SALESAMOUNT'], fit=norm)


# In[68]:


#Get the QQ-plot
fig = plt.figure()
res = stats.probplot(df['SALESQUANTITY'], plot=plt)
plt.show()


# In[69]:


#The data is highly skewed, but since we'll be applying ARIMA, it's fine.
df['SALESQUANTITY'].skew()


# In[70]:


#The amount of orders shipped by each country.
df.groupby('date_col').sum().sort_values('date_col', ascending = True)


# In[71]:


#Summation by CountryName
df.groupby('CountryName').sum().sort_values('SALESQUANTITY',ascending = False)
# we can see that China has highest sales quantity, followed by Germanu and UK


# In[72]:


#Product Category.

print (len(df['CountryName'].value_counts()))

rcParams['figure.figsize'] = 50,14
sns.countplot(df['CountryName'].sort_values(ascending = True))

#We can see that number of orders is highest in China and so is the sum of sales


# In[73]:


#Lets check the orders by CountryName.

#Checking with Boxplots
# figure size in inches
rcParams['figure.figsize'] = 16,4
f, axes = plt.subplots(1, 2)
#Regular Data
fig1 = sns.boxplot( df['CountryName'],df['SALESQUANTITY'], ax = axes[0])
#Data with Log Transformation
fig2 = sns.boxplot( df['CountryName'], np.log1p(df['SALESQUANTITY']),ax = axes[1])


# In[74]:


#Lets check the Orders by ContinentName.
rcParams['figure.figsize'] = 50,20
#Taking subset of data temporarily for in memory compute.
df_temp = df.sample(n=20000).reset_index()
fig3 = sns.boxplot( df_temp['ContinentName'].sort_values(),np.log1p(df_temp['SALESQUANTITY']))
del df_temp, fig3


# In[75]:


# Time Series Analysis
df1 = df.groupby('date_col')['SALESQUANTITY'].sum().reset_index()
#Index the date
df = df.set_index('date_col')
df.index #Lets check the index
#This gives us the total orders placed on each day and loads the data in df1 data frame.
df1
ax = sns.lineplot(x="date_col", y="SALESQUANTITY", data=df1)
ax.plot(figsize=(12,5))
plt.show()


# In[76]:


y = df['SALESQUANTITY'].resample('MS').mean()
y.plot(figsize=(12,5))
plt.show()


# In[47]:


#The best part about time series data and decomposition is that you can break down the data into the following:
#Time Series Decomposition. 
from pylab import rcParams
import statsmodels.api as sm
rcParams['figure.figsize'] = 18, 8
decomposition = sm.tsa.seasonal_decompose(y, model='additive')
fig = decomposition.plot()
plt.show()

#Takeaway: The sales are always highest for the beginning of the year and the lowest demand every year is in the
#last quarter. The observed trend shows that orders were higher during 2009 and have been increasing continiously.


# In[77]:


#Grid Search

#Since ARIMA has hyper params that can be tuned, the objective here is to find the best params using Grid Search.

#Sample params for seasonal arima. (SARIMAX).

#For each combination of parameters, we fit a new seasonal ARIMA model with the SARIMAX() function 
#from the statsmodels module and assess its overall quality.

import itertools
p = d = q = range(0, 2)
pdq = list(itertools.product(p, d, q))
seasonal_pdq = [(x[0], x[1], x[2], 12) for x in list(itertools.product(p, d, q))]
print('Examples of parameter combinations for Seasonal ARIMA...')
print('SARIMAX: {} x {}'.format(pdq[1], seasonal_pdq[1]))
print('SARIMAX: {} x {}'.format(pdq[1], seasonal_pdq[2]))
print('SARIMAX: {} x {}'.format(pdq[2], seasonal_pdq[3]))
print('SARIMAX: {} x {}'.format(pdq[2], seasonal_pdq[4]))


# In[78]:


#Get the best params for the data. Choose the lowest AIC.

# The Akaike information criterion (AIC) is an estimator of the relative quality of statistical models for a 
# given set of data. 
# AIC measures how well a model fits the data while taking into account the overall complexity of the model.
# Large AIC: Model fits very well using a lot of features.
# Small AIC: Model fits similar fit but using lesser features. 
# Hence LOWER THE AIC, the better it is.

#The code tests the given params using sarimax and outputs the AIC scores.

for param in pdq:
    for param_seasonal in seasonal_pdq:
        try:
            mod = sm.tsa.statespace.SARIMAX(y,
                                            order=param,
                                            seasonal_order=param_seasonal,
                                            enforce_stationarity=False,
                                            enforce_invertibility=False)

            results = mod.fit()

            print('SARIMA{}x{}12 - AIC:{}'.format(param, param_seasonal, results.aic))
        except:
            continue


# In[79]:


#Fit the model with the best params.
#ARIMA(0, 0, 1)x(1, 1, 1, 12)12 - AIC:8.89680937464163


#The above output suggests that ARIMA(0, 0, 1)x(1, 1, 1, 12) yields the lowest AIC value: 8.89
#Therefore we should consider this to be optimal option.

from statsmodels.tsa.statespace.sarimax import SARIMAX
mod = sm.tsa.statespace.SARIMAX(y,
                                order=(0, 0, 1),
                                seasonal_order=(1, 1, 1, 12),
                                enforce_stationarity=False,
                                enforce_invertibility=False)
results = mod.fit()
print(results.summary().tables[1])


# In[80]:


#Interpreting the table:

#coeff: Shows weight/impotance how each feature impacts the time series. Pvalue: Shows the significance of each feature weight. Can test hypothesis using this. If p value is <.05 then they are statitically significant.

#Refresher on null hyp and pvalues. By default we take the null hyp as 'there is no relationship bw them' If p value < .05 (significance level) then you reject the Null Hypthesis If p value > .05 , then you fail to reject the Null Hypothesis.

#So, if the p-value is < .05 then there is a relationship between the response and predictor. Hence, significant.


#Plotting the diagnostics.

#The plot_diagnostics object allows us to quickly generate model diagnostics and investigate for any unusual behavior.
results.plot_diagnostics(figsize=(16, 8))
plt.show()

#What to look for?
#1. Residuals SHOULD be Normally Distributed ; Cross
#Hence, we cannot perform seasonal ARIMA on this data.


# In[ ]:




