"""File to copy the LCPS ICU admission programme"""
import numpy as np
import pandas as pd
import cvxpy as cp
from sklearn.model_selection import cross_val_score

# load data
master = pd.read_csv("Data/master.csv")
master.date = pd.to_datetime(master.date, format='%Y-%m-%d')
master = master.sort_values(by='date', ascending=True)

# create weekday variable.
# 0 is Monday, ..., 6 is Sunday
master['weekday'] = master.date.dt.weekday

y = master.ICU_Inflow
w = master.weekday
# Split data set into testing and training set. 80% in training set (arbitrary
# choice)

# dit omdraaien bij nieuwe load_data
y_train = y[:int(y.shape[0]*0.7)]
y_test = y[int(y.shape[0]*0.7):]
w_train = w[:int(y.shape[0]*0.7)]
w_test = w[int(y.shape[0]*0.7):]


class LCPS:

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
        w_pred = w_train.iloc[-7 + (t-1)]

        return np.exp(x[-1] + t * (x[-1] - x[-2]) + s[w_pred])

    def mse(self, y):
        return None

    def solve(self):
        p = self.y.shape
        x = cp.Variable(p)
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
    for t in range(1, len(y_test)):
        pred_y.append(algo.predict(algo.x, algo.s, w_train, t=t))

    print(np.array(pred_y))


test(LCPS)

"""
algo = LCPS(y, gamma=1)
algo.solve()

algo = LCPS(y=master.ICU_Inflow, weekday=master.weekday, gamma=1)
x = np.random.poisson(lam=20, size=len(algo.y))
s = np.random.randn(7)
algo.loss(x, s)
"""