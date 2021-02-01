#!/usr/bin/env python
# coding: utf-8

# In[145]:





# Getting information about the stock

# In[1]:


import yfinance as yf
import pandas as pd
from datetime import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.arima_model import ARIMA
from sklearn.metrics import mean_squared_error
from statsmodels.tsa.statespace.sarimax import SARIMAX
import pickle


# In[2]:


#setting input parameters

stockName = input('get names from stocks from here https://www.nasdaq.com/market-activity/stocks/screener:')
startDate = input('Please enter in yyyymmdd format: ')
endDate = input('Please enter in yyyymmdd format: ')


# def stockData(stockName, startDate, endDate):
#    stockName = yf.Ticker(stockName)
#    startDate = datetime.strptime(startDate, '%Y%m%d').strftime('%Y-%m-%d')
#    endDate = datetime.strptime(endDate, '%Y%m%d').strftime('%Y-%m-%d')
#    data = stockName.history(period = 'ld', start = startDate, end = '2021-01-21')
#    return data


# In[2]:


stockName = yf.Ticker('AMZN')
initalData = stockName.history(period = 'ld', start = '2015-01-01', end = '2021-01-21')


# In[3]:


type(initalData)


# In[4]:


data = initalData[['Open', 'High', 'Low', 'Close', 'Volume']]
data.head()


# In[5]:


def myBoxPlot():
    plt.style.use('seaborn')
    for column in data.columns:
        plt.boxplot(data[column])
        plt.xlabel(column)
        plt.show()   
    
myBoxPlot()


# In[6]:


import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

plt.style.use('seaborn')
graphOne = plt.boxplot([data[graph] for graph in data.columns])


# In[7]:


def myLinePlot():
    cols = ['High', 'Low', 'Open', 'Close']
    color = ['c','m','y','k']
    i = 0
    for column in cols:
        plt.plot(data.index.to_pydatetime(), data[column], color=color[i], label=column)
        i+=1
        plt.xlabel('Date')
        plt.legend()
        plt.title('Amazon Stock Price' )
        
        
myLinePlot()


# In[8]:


#rolling data to make it look Clean

def myClearLinePlot():
    dataClear = data.rolling(window=50).mean()
    cols = ['High', 'Low', 'Open', 'Close']
    color = ['c','m','y','k']
    i = 0
    for column in cols:
        plt.plot(dataClear.index.to_pydatetime(), dataClear[column], color=color[i], label=column)
        i+=1
        plt.xlabel('Date')
        plt.legend()
        
myClearLinePlot()
    


# In[9]:


#resampling data to aggregate by month
def resamplingData():
    resampledData = data['Close'].resample('MS').mean()
    return resampledData

dataByMonth = resamplingData()
dataByMonth.head()


# In[10]:


seasonalPlot = seasonal_decompose(dataByMonth,model='add')
seasonalPlot.plot()
plt.show()


# In[11]:


def adfullerTest(column):
    result = adfuller(column)
    labels = ['ADF Test Statistic','p-value','#Lags used','Number of observations']
    for value,label in zip(result,labels):
        print(label+' :'+str(value))
    print('Critical Values:')
    for key,value in result[4].items():
        print(key+' : '+str(round(value,6)))
    print('')
    if result[1] <= 0.05:
        print('Data is stationary')
    else:
        print('Data is non-stationary')


# In[12]:


adfullerTest(dataByMonth)


# In[13]:


differencingData = dataByMonth - dataByMonth.shift(1)

print(differencingData)

adfullerTest(differencingData.dropna())


# In[14]:


differencingDataByMonth = dataByMonth - dataByMonth.shift(12)


# In[15]:


print(adfullerTest(differencingData.dropna()))


# In[16]:


plot_acf(dataByMonth)


# In[17]:


plot_pacf(dataByMonth)


# In[18]:


dataByMonth.count()


# In[65]:


modelData = dataByMonth
train = modelData[:53]
test = modelData[53:]


# In[68]:


train.head()


# In[69]:


modelFit = ARIMA(train, order=(2, 0, 2)).fit()


# In[70]:


modelFit.summary()


# In[71]:


forcastData = modelFit.forecast(steps=20)[0]

meanSquareError = mean_squared_error(test, forcastData)
print('MSE: '+str(meanSquareError))
rootMeanSquareError = np.sqrt(meanSquareError)
print('RMSE: '+str(rootMeanSquareError))


# In[72]:


plt.figure(figsize=(12,5))
plt.plot(train.index.to_pydatetime(), train, label='training')
plt.plot(test.index.to_pydatetime(), test, label='actual')
plt.plot(test.index.to_pydatetime(), forcastData, label='forecast')
plt.legend()


# In[73]:


sarimaxModel = SARIMAX(train, order=(1,1,4), seasonal_order=(0,1,2,12))
sarimaxModelFit = sarimaxModel.fit()


# In[74]:


sarimaxModelFit.summary()


# In[75]:


sarimaxForecast = sarimaxModelFit.forecast(steps=20)


# In[76]:


meanSquareError = mean_squared_error(test, sarimaxForecast)
print('MSE: '+str(meanSquareError))
rootMeanSquareError = np.sqrt(meanSquareError)
print('RMSE: '+str(rootMeanSquareError))


# In[77]:


plt.figure(figsize=(12,5))
plt.plot(train.index.to_pydatetime(), train, label='training')
plt.plot(test.index.to_pydatetime(), test, label='actual')
plt.plot(test.index.to_pydatetime(), sarimaxForecast, label='forecast')
plt.legend()


# In[78]:


sarimaxOnFullData = SARIMAX(modelData, order=(1,1,4), seasonal_order=(0,1,2,12))
sarimaxFullModelFit = sarimaxOnFullData.fit()


# In[79]:


sarimaxFullModelFit.summary()


# In[80]:


sarimaxFullModelFit.plot_diagnostics()


# In[102]:


futurePredict = sarimaxFullModelFit.get_prediction(start='2021-01-25',end='2022-12-01')


# In[107]:


futurePredict.conf_int()


# In[99]:


pred_ci = futurePredict.conf_int()


# In[100]:


def finalPrediction():
    plt.figure(figsize=(12,5))
    plt.plot(modelData.index.to_pydatetime(), modelData, label='training')
    futurePredict.predicted_mean.plot()
    plt.fill_between(pred_ci.index, pred_ci.iloc[:, 0], pred_ci.iloc[:, 1], color='k', alpha=.2)
    plt.legend()
    
    
finalPrediction()


# In[84]:


pickle.dump(sarimaxFullModelFit, open('model.pkl', 'wb'))


# In[85]:


model = pickle.load(open('model.pkl', 'rb'))


# In[1]:


prediction = model.predict(start = '2021-01-25', end = '2022-12-01', )

