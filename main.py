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