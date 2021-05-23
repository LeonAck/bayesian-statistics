"""File to copy the LCPS ICU admission programme"""
import numpy as np
from numpy import linalg as LA
import cvxpy as cp
import matplotlib.pyplot as plt

class Predictor:

    def __init__(self, X, Y, gamma=0):
        self.X = X
        self.Y = Y
        self.gamma = gamma
        self.beta_0 = 0

    def loss(self, X, Y, beta):
        # absolute value loss function for michiel
        return np.square(Y - X @ beta).sum()

    def solve(self):
        raise NotImplemented

    def predict(self, x):
        return self.beta_0 + x @ self.beta

    def mse(self, X, Y):
        n, p = X.shape
        return self.loss(X, Y, self.beta).value / n


class Ridge_convex(Predictor):
    def loss(self, X, Y, beta):
        return cp.pnorm(Y - X @ beta, p=2) ** 2

    def regularizer(self, beta):
        return cp.pnorm(beta, p=2) ** 2

    def objective(self, beta):
        return self.loss(self.X, self.Y, beta) + self.gamma * self.regularizer(beta)

    def solve(self):
        n, p = self.X.shape
        beta = cp.Variable(p)
        obj = cp.Minimize(self.objective(beta))
        problem = cp.Problem(obj)
        problem.solve(solver="SCS")
        self.beta = np.array(beta.value)


class LCPS:
    def __init__(self, y, gamma=0, t=1):
        self.y = y
        self.gamma = gamma
        self.t = t

    def regularizer(self, x):
        """
        Penalty term that penalizes trend changes
        """
        return sum((x[3:] - x[2:-1]) - (x[2:-1] - x[1:-2]))

    def predict(self, x):
        """
        Function to get the t-day ahead prediction
        """
        # week day nog toevoegen
        return np.exp(x[1:] + self.t * (x[1:] - x[:-1]))

    def loss(self, x):
        return sum(np.abs(x - np.log(self.y)))

    def objective(self, x):
        return self.loss(self.y) + self.gamma * self.regularizer(x)

    def solve(self):
        n, p = self.X.shape
        beta = cp.Variable(p)
        obj = cp.Minimize(self.objective(beta))
        problem = cp.Problem(obj)
        problem.solve(solver="SCS")
        self.beta = np.array(beta.value)
