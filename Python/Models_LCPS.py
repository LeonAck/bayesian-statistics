"""File to copy the LCPS ICU admission programme"""
import numpy as np
import pandas as pd
import cvxpy as cp

# load data
master = pd.read_csv("Data/master.csv")
master.date = pd.to_datetime(master.date, format='%Y-%m-%d')

# create weekday variable.
# 0 is Monday, ..., 6 is Sunday
master['weekday'] = master.date.dt.weekday

y = master.ICU_Inflow
w = master.weekday
# Split data set into testing and training set. 80% in training set (arbitrary
# choice)
y_train = y[int(y.shape[0]*0.2):]
y_test = y[:int(y.shape[0]*0.2)]
w_train = w[int(y.shape[0]*0.2):]
w_test = w[:int(y.shape[0]*0.2)]


class LCPS:
    def __init__(self, y, w=None, gamma=0):
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
        return sum(cp.abs((x[3:] - x[2:-1]) - (x[2:-1] - x[1:-2])))

    def objective(self, x, s):
        return self.loss(x, s) + self.gamma * self.regularizer(x)

    def predict(self, x, s, w, t):
        """
        Function to get the t-day ahead prediction
        """
        return np.exp(x[1:] + t * (x[1:] - x[:-1]) + s[w[t:]])

    def mse(self, y):
        return None

    def solve(self):
        p = self.y.shape
        x = cp.Variable(p)
        s = cp.Variable((7,))
        obj = cp.Minimize(self.objective(x, s))
        problem = cp.Problem(obj)
        problem.solve('ECOS')
        self.x = np.array(x.value)
        self.s = np.array(s.value)


def test(method):

    algo = method(y_train, w_train, gamma=1)
    algo.solve()
    print("s", algo.s)
    print("algo_x", algo.x)
    print("y", np.log(y_train))

    d = {'x': algo.x, 'weekday': w_train}
    #data_pred = pd.DataFrame(data=d)
    algo.predict(algo.x, algo.s, master.weekday, t=1)


test(LCPS)
"""
algo = LCPS(y, gamma=1)
algo.solve()

algo = LCPS(y=master.ICU_Inflow, weekday=master.weekday, gamma=1)
x = np.random.poisson(lam=20, size=len(algo.y))
s = np.random.randn(7)
algo.loss(x, s)
"""