#messing around with matplotlib for a power consumption data set
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()
from sklearn.preprocessing import LabelEncoder

power_data_path = (r'C:\Users\Ernest Fedorowich\Documents\projects\AEP_hourly.csv')
power_data = pd.read_csv(power_data_path, parse_dates=['Datetime'])

power_data = power_data.assign(hour = power_data.Datetime.dt.hour,
                               day = power_data.Datetime.dt.day,
                               month = power_data.Datetime.dt.month,
                               year = power_data.Datetime.dt.year)

Datetime = pd.Series(power_data.index, index=power_data.Datetime, name="Count_1_day").sort_index()

count_1_day = Datetime.rolling('1d').count()

plt.plot(count_1_day[22:]);
plt.title("Power consumption in the last day");


count_1_day.index = Datetime.values
count_1_day = count_1_day.reindex(power_data.index)

max_value = power_data['AEP_MW'].max()
print(max_value)
