#!/usr/bin/env python
# coding: utf-8

# In[2]:


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


# In[3]:


df  = pd.read_csv('RoomsDemand.csv',parse_dates = {'Date' : [ "Month", "Year"]})
pd.set_option('display.float_format', lambda x: '%.3f' % x)


# In[4]:


df.head()


# In[5]:


# Check any number of columns with NaN
print(df.isnull().any().sum(), ' / ', len(df.columns))
# Check any number of data points with NaN
print(df.isnull().any(axis=1).sum(), ' / ', len(df))


# In[6]:


#Lets check where these nulls are.
print (df.isna().sum())
print ('Null to Dataset Ratio in Dates: ',df.isnull().sum()[3]/df.shape[0]*100)
#There are missing values in Dates.


# In[7]:


df.dtypes


# In[8]:


#checking number of columns and rows
df.shape


# In[9]:


# Check any number of columns with NaN
print(df.isnull().any().sum(), ' / ', len(df.columns))
# Check any number of data points with NaN
print(df.isnull().any(axis=1).sum(), ' / ', len(df))


# In[10]:


#seems like we dont have any column or row with null value
# lets have a look at distribution plot of Order_Demand
sns.distplot(df['Demand'], fit=norm)


# In[11]:


#Get the QQ-plot
fig = plt.figure()
res = stats.probplot(df['Demand'], plot=plt)
plt.show()


# In[12]:


#The data is less skewed but is bimodal and not Normally distributed. However, since we are running ARIMA it should be fine
df['Demand'].skew()


# In[14]:


#Number of data points by each country.
df['Country'].value_counts().sort_values(ascending = False)


# In[15]:


#Demand of rooms created by each country.
df.groupby('Country').sum().sort_values('Demand', ascending = False)


# In[16]:


#Lets check the orders by warehouse.

#Checking with Boxplots
from matplotlib import rcParams
# figure size in inches
rcParams['figure.figsize'] = 16,4
f, axes = plt.subplots(1, 2)
#Regular Data
fig3 = sns.boxplot( df['Country'],df['Demand'], ax = axes[0])
#Data with Log Transformation
fig4 = sns.boxplot( df['Country'], np.log1p(df['Demand']),ax = axes[1])

del fig3, fig4


# In[19]:


df = df.groupby('Date')['Demand'].sum().reset_index()
#This gives us the total orders placed on each day.


# In[20]:


#Index the date
df = df.set_index('Date')
df.index #Lets check the index

y = df['Demand'].resample('MS').mean()
y.plot(figsize=(12,5))
plt.show()


# In[21]:


#The best part about time series data and decomposition is that you can break down the data into the following:
#Time Series Decomposition. 
from pylab import rcParams
import statsmodels.api as sm
rcParams['figure.figsize'] = 18, 8
decomposition = sm.tsa.seasonal_decompose(y, model='additive')
fig = decomposition.plot()
plt.show()

#Takeaway: The rooms bookings are always lowest at the starting and ending of every year. 
#The observed trend shows that number of rooms booked have been increasing continiously since 2012.


# In[22]:


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


# In[23]:


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


# In[24]:


#Fit the model with the best params.
#ARIMA(1, 1, 1)x(1, 1, 1, 12)12 - AIC:1924.0346632592104


#The above output suggests that ARIMA(1, 1, 1)x(1, 1, 1, 12) yields the lowest AIC value: 1924.0346632592104
#Therefore we should consider this to be optimal option.

from statsmodels.tsa.statespace.sarimax import SARIMAX
mod = sm.tsa.statespace.SARIMAX(y,
                                order=(1, 1, 1),
                                seasonal_order=(1, 1, 1, 12),
                                enforce_stationarity=False,
                                enforce_invertibility=False)
results = mod.fit()
print(results.summary().tables[1])


# In[25]:


#Interpreting the table:

#coeff: Shows weight/impotance how each feature impacts the time series. Pvalue: Shows the significance of each feature weight. Can test hypothesis using this. If p value is <.05 then they are statitically significant.

#Refresher on null hyp and pvalues. By default we take the null hyp as 'there is no relationship bw them' If p value < .05 (significance level) then you reject the Null Hypthesis If p value > .05 , then you fail to reject the Null Hypothesis.

#So, if the p-value is < .05 then there is a relationship between the response and predictor. Hence, significant.


#Plotting the diagnostics.

#The plot_diagnostics object allows us to quickly generate model diagnostics and investigate for any unusual behavior.
results.plot_diagnostics(figsize=(16, 8))
plt.show()

#What to look for?
#1. Residuals SHOULD be Normally Distributed ; Check
#Top Right: The (orange colored) KDE line should be closely matched with green colored N(0,1) line. This is the standard notation
#for normal distribution with mean 0 and sd 1.
#Bottom Left: The qq plot shows the ordered distribution of residuals (blue dots) follows the linear trend of the samples 
#taken from a standard normal distribution with N(0, 1). 

#2. #Residuals are not correlated; Check
#Top Left: The standard residuals don’t display any obvious seasonality and appear to be white noise. 
#Bottom Right: The autocorrelation (i.e. correlogram) plot on the bottom right, which shows that the time series residuals have 
#low correlation with its own lagged versions.


# In[31]:


#Lets get the predictions and confidence interval for those predictions.
#Get the predictions. The forecasts start from the 1st of Aug 2014 but the previous line shows how it fits to the data.
pred = results.get_prediction(start=pd.to_datetime('2014-08-01'), dynamic=False) #false is when using the entire history.
#Confidence interval.
pred_ci = pred.conf_int()

#Plotting real and forecasted values.
ax = y['2013':].plot(label='observed')
pred.predicted_mean.plot(ax=ax, label='One-step ahead Forecast', alpha=.7, figsize=(14, 7))
ax.fill_between(pred_ci.index,
                pred_ci.iloc[:, 0],
                pred_ci.iloc[:, 1], color='blue', alpha=.2)
ax.set_xlabel('Date')
ax.set_ylabel('Demand')
plt.legend()
plt.show()

#Takeaway: The forecats seems to be fitting well to the data. The Blue/purple thicker plot shows the confidence level in the forecasts. 


# In[32]:


#Getting the mean squared error (average error of forecasts).
y_forecasted = pred.predicted_mean
y_truth = y['2016-01-01':]
mse = ((y_forecasted - y_truth) ** 2).mean()
print('MSE {}'.format(round(mse, 2)))

#Smaller the better.


# In[33]:


print('RMSE: {}'.format(round(np.sqrt(mse), 2)))


# In[34]:


#The time can be changed using steps.
pred_uc = results.get_forecast(steps=50)
pred_ci = pred_uc.conf_int()
ax = y.plot(label='observed', figsize=(14, 7))
pred_uc.predicted_mean.plot(ax=ax, label='Forecast')
ax.fill_between(pred_ci.index,
                pred_ci.iloc[:, 0],
                pred_ci.iloc[:, 1], color='k', alpha=.25)
ax.set_xlabel('Date')
ax.set_ylabel('Demand')
plt.legend()
plt.show()

#Far out values are naturally more prone to variance. The grey area is the confidence we have in the predictions.


# In[ ]:




