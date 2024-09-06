import pandas as pd
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import datetime as dt

#import data and convert into pandas dataframe object
stock_data = pd.read_csv('NVDA.csv' , index_col='Date')

#create a figure to plot onto
plt.figure(figsize=(15,10))
#format date strings into datetime format
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=60))
#create array of dates
x_dates = [dt.datetime.strptime(d,'%Y-%m-%d').date() for d in stock_data.index.values]
#plot High and Low prices against date
plt.plot( x_dates, np.array(stock_data['High']), label='High')
plt.plot( x_dates, np.array(stock_data['Low']), label='Low')
plt.xlabel('Date')
plt.ylabel('Stock Price (USD)')
plt.grid()
plt.legend()
plt.show()

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import *
from tensorflow.keras.callbacks import EarlyStopping

from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.model_selection import train_test_split
from sklearn.model_selection import TimeSeriesSplit

target_y = stock_data['Close']
x_feat = stock_data.iloc[:, 0:3]

sc = StandardScaler()
X_ft = sc.fit_transform(x_feat.values)
X_ft = pd.DataFrame(columns=x_feat.columns, data=X_ft, index=x_feat.index)

def lstm_split(data, n_steps):
    X, y = [], []
    for i in range(len(data) - n_steps+1):
        X.append(data[i:i+n_steps, :-1])
        y.append(data[i+n_steps-1, -1])

    return np.array(X), np.array(y)

X1, y1 = lstm_split(stock_data_ft