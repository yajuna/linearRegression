import re
import pandas as pd
import time

# start timer
start = time.time()


data = pd.read_csv('data/combined_data.csv')
data['north'] = [1 if d == 'N' else 0 for d in data['direction']]
data['south'] = [1 if d == 'S' else 0 for d in data['direction']]
data['west'] = [1 if d == 'W' else 0 for d in data['direction']]
data['east'] = [1 if d == 'E' else 0 for d in data['direction']]
data['second'] = [re.split("[- :]", h)[5] for h in data['Date Time']]
data['minute'] = [re.split("[- :]", h)[4] for h in data['Date Time']]
data['hour'] = [re.split("[- :]", h)[3] for h in data['Date Time']]
data['day'] = [re.split("[- :]", h)[2] for h in data['Date Time']]
data['month'] = [re.split("[- :]", h)[1] for h in data['Date Time']]
data['year'] = [re.split("[- :]", h)[0] for h in data['Date Time']]


data = data.drop('direction', axis=1)
data = data.drop('Date Time', axis=1)
print(data.head())
data.to_csv('~/UW/code/TMATH495/data/temperature_features.csv')


end = time.time()

print(data.head())
print(end-start)

