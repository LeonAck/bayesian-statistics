import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys

sys.path.append("Python")
from cross_validation import BlockingTimeSeriesSplitLCPS
from Models_LCPS import gridsearch_lcps, rolling_pred_testset

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

# define the grid
grid = np.arange(0, 101, 0.5)


opt_lambda, average_mae_per_par = gridsearch_lcps(y, w, splits_list, grid=grid)
print(opt_lambda)

filename = 'opt_lambda_LCPS.txt'
with open(filename, 'w') as file_object:
    file_object.write(str(opt_lambda))
    file_object.write(str(average_mae_per_par))

"""
y_pred = rolling_pred_testset(LCPS, y_train, y_test, w_train, w_test, t=1)


filename = 'y_pred_rolling_LCPS.txt'
with open(filename, 'w') as file_object:
    file_object.write(str(y_pred))
"""

# load LCPS_rolling one-day predictions
my_file = "y_pred_rolling_LCPS.txt"

with open(my_file, 'r') as f:
    yhat_lcps_oneday = eval(f.read())


# Graph of predictions LCPS
plt.plot(data.index[int(y.shape[0] * split_pct):], y_test, label='ICU Admissions')
plt.plot(data.index[int(y.shape[0] * split_pct):], yhat_lcps_oneday, label='Predictions')
plt.xlabel('Time')
plt.ylabel('Admissions')
plt.legend()
plt.show()
