"""File to copy the LCPS ICU admission programme"""
import cvxpy as cp
import numpy as np
from sklearn.metrics import mean_absolute_error


class LCPSModel:
    """
    Class to recreate LCPS model
    Minimization with trend penalty term
    """

    def __init__(self, y, w, gamma=10):
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


def rolling_pred_LCPS(method, y_train, y_test, w_train, w_test, t=1, gamma=10):
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
        algo = method(y_train, w_train, gamma=gamma)

        # solve the model
        algo.solve()

        # add prediction to list of predictions
        y_pred.append(algo.predict(algo.x, algo.s, w_train, t=t))

    return y_pred


def gridsearch_LCPS(y, w, splits_list, grid=None, t=1):
    """
    Find the optimal value for the smoothing parameter lambda by a block-
    time series split. We optimzie based on the mean absolute error of rolling
    predictions

    :return:
    optimal value of lambda
    """
    # repeat loop for every parameter in grid
    average_mae_per_par = dict()
    for parameter in grid:

        mae_list = []

        # for loop for each set of indices per fold
        for index_dict in splits_list:
            # perform rolling predictions using train set on the validation set
            y_pred = rolling_pred_LCPS(LCPSModel,
                                          y[index_dict["train"][0]:
                                            index_dict["train"][1]],
                                          y[index_dict["validation"][0]:
                                            index_dict["validation"][1]],
                                          w[index_dict["train"][0]:
                                            index_dict["train"][1]],
                                          w[index_dict["validation"][0]:
                                            index_dict["validation"][1]],
                                       t=t, gamma=parameter)

            # add the mean absolute error on validation set to the list
            mae_list.append(mean_absolute_error(
                np.exp(y[
                       index_dict["validation"][0]:
                       index_dict["validation"][1]]),
                np.exp(y_pred)))

        # add average mae for parameter to dict
        average_mae_per_par["{}".format(parameter)] = np.mean(mae_list)

    # return parameter with average mae

    return min(average_mae_per_par, key=average_mae_per_par.get), \
           average_mae_per_par


