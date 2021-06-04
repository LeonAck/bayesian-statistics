
"""
File that runs LCPS model
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys

sys.path.append("Python")
from cross_validation import BlockingTimeSeriesSplitLCPS
from Models_LCPS import gridsearch_LCPS, rolling_pred_LCPS, LCPSModel

# load data
data = pd.read_csv("Data/data.csv")
data.date = pd.to_datetime(data.date, format='%Y-%m-%d')
data = data.sort_values(by='date', ascending=True)

# create weekday variable.
# 0 is Monday, ..., 6 is Sunday
data['weekday'] = data.date.dt.weekday

y = data.ICU_Inflow.values
w = data.weekday.values

# Split data set into testing and training set. 80% in training set (arbitrary
# choice)

split_pct = 0.8
y_train = y[:int(y.shape[0] * split_pct)]
y_test = y[int(y.shape[0] * split_pct):]
w_train = w[:int(y.shape[0] * split_pct)]
w_test = w[int(y.shape[0] * split_pct):]


btscv = BlockingTimeSeriesSplitLCPS(n_splits=5)

# splits_list is a list of dictionaries, where each dictionary is a fold
# the dictionary contains the start and stop indices for the train set
# and the start and stop indices for the validation set
splits_list = btscv.return_split(y_train)

# first rough search for the smoothing parameter using blocktimeseries split
grid_1 = np.arange(0, 101, 0.5)
opt_lambda_1, average_mae_per_par_1 = gridsearch_LCPS(y, w, splits_list, grid=grid_1)

print(opt_lambda_1)

# Plot graph to inspect where low points of the mae are
plt.plot(grid_1, average_mae_per_par_1.values(),  label='MAE')
plt.xlabel('Lambda')
plt.ylabel('MAE')
axes = plt.gca()
axes.set_xlim([0, 100])
plt.legend()
plt.show()

# More sophisticated search
# define the grid
grid_2 = np.arange(3, 10, 0.01)

opt_lambda_2, average_mae_per_par_2 = gridsearch_LCPS(y, w, splits_list, grid=grid_2)

print(opt_lambda_2)

# compute rolling predictions based on
y_pred = rolling_pred_LCPS(LCPSModel, y_train, y_test, w_train, w_test, t=1, gamma=opt_lambda_2)

# save predictions to text file
filename = 'y_pred_rolling_LCPS.txt'
with open(filename, 'w') as file_object:
    file_object.write(str(y_pred))
