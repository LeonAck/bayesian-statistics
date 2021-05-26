"""File to copy the LCPS ICU admission programme"""
import numpy as np
import pandas as pd
import cvxpy as cp
import matplotlib.pyplot as plt

import sys

sys.path.append("Python")
from cross_validation import perf_metrics

# load data
master = pd.read_csv("Data/master.csv")
master.date = pd.to_datetime(master.date, format='%Y-%m-%d')
master = master.sort_values(by='date', ascending=True)

# create weekday variable.
# 0 is Monday, ..., 6 is Sunday
master['weekday'] = master.date.dt.weekday

y = master.ICU_Inflow.values
w = master.weekday.values
# Split data set into testing and training set. 80% in training set (arbitrary
# choice)

# dit omdraaien bij nieuwe load_data
split_pct = 0.8
y_train = y[:int(y.shape[0] * split_pct)]
y_test = y[int(y.shape[0] * split_pct):]
w_train = w[:int(y.shape[0] * split_pct)]
w_test = w[int(y.shape[0] * split_pct):]


class LCPS:
    """
    Class to recreate LCPS model
    Minimization with trend penalty term

    """

    def __init__(self, y, w=None, gamma=10):
        self.y = y
        self.gamma = gamma
        self.w = w

    def loss(self, x, s):
        """Function for loss function"""
        return sum(cp.abs(x + s[self.w] - np.log(self.y)))

    def regularizer(self, x):
        """
        Penalty term that penalizes trend changes
        """
        return sum(cp.abs((x[2:] - x[1:-1]) - (x[1:-1] - x[:-2])))

    def objective(self, x, s):
        return self.loss(x, s) + self.gamma * self.regularizer(x)

    def predict(self, x, s, w_train, t):
        """
        Function to get the t-day ahead prediction
        """
        # w_pred is the weekday of the day we want to predict. Given the weekday
        # of x[-1]
        w_pred = w_train[-7 + (t - 1)]

        return np.exp(x[-1] + t * (x[-1] - x[-2]) + s[w_pred])

    def solve(self):
        p = self.y.shape
        x = cp.Variable(p)
        # variable for days of the week
        s = cp.Variable((7,))
        obj = cp.Minimize(self.objective(x, s))
        problem = cp.Problem(obj)
        # different solver?
        problem.solve('ECOS')
        self.x = np.array(x.value)
        self.s = np.array(s.value)


def test(method):
    algo = method(y_train, w_train, gamma=10)
    algo.solve()
    print("s", algo.s)
    print("algo_x", algo.x)
    print("y", np.log(y_train))

    pred_y = []
    for t in range(1, len(y_test) + 1):
        pred_y.append(algo.predict(algo.x, algo.s, w_train, t=t))

    # exp back to log to insert into formula
    print(perf_metrics(np.log(y_test), np.log(pred_y)))


# test(LCPS)


def rolling_pred(method, y, w, t):
    """
    Rolling predctions for model. Uses data up to one day To predict t-day
    prediction. Then moves up one day to and predicts t-day from that day
    """
    y_pred = []

    for i in range(6, len(y) - 7):
        y_train = y[:i]
        w_train = w[:i]
        print(i, y_train)
        algo = method(y_train, w_train, gamma=10)
        algo.solve()

        y_pred.append(algo.predict(algo.x, algo.s, w_train, t=t))
    print(y_pred)
    return y_pred


def rolling_pred_testset(method, y_train, y_test, w_train, w_test, t=1):
    """
    Function to perform a rolling prediction for values in the test set.
    Model is first estimated on training set, but data points from the test
    set are added iteratively.
    """

    # create list for rolling predictions
    y_pred = []

    # we make a prediction for every element in the test set
    for i in range(len(y_test)):

        # for i = 0, we make a prediction on the training set
        # for i > 0, we add the next observation to the training set
        if i > 0:
            y_train = np.append(y_train, y_test[i - 1])
            w_train = np.append(w_train, w_test[i - 1])

        # we create a model based on the training and test set
        algo = method(y_train, w_train, gamma=10)

        # solve the model
        algo.solve()

        # add prediction to list of predictions
        y_pred.append(algo.predict(algo.x, algo.s, w_train, t=t))

    return y_pred

"""
# y_pred = rolling_pred_testset(LCPS, y_train, y_test, w_train, w_test, t=1)
# y_pred = rolling_pred(LCPS, y=y, w=w, t=1)

filename = 'y_pred_rolling_LCPS.txt'
with open(filename, 'w') as file_object:
    file_object.write(str(y_pred))
"""

# load LCPS_rolling one-day predictions
my_file = "y_pred_rolling_LCPS.txt"

with open(my_file, 'r') as f:
    yhat_lcps_oneday = eval(f.read())
# Graph of predictions
plt.plot(master.index[int(y.shape[0] * split_pct):], y_test, label='ICU Admissions')
plt.plot(master.index[int(y.shape[0] * split_pct):], yhat_lcps_oneday, label='Predictions')
plt.xlabel('Time')
plt.ylabel('Admissions')
plt.legend()
plt.show()
