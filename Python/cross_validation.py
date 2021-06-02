"""
Code from https://hub.packtpub.com/cross-validation-strategies-for-time-series-forecasting-tutorial/
To make BlockingTimeSeriesSplit
"""
import numpy as np
from sklearn.model_selection import cross_val_score
import sklearn.metrics as metrics


class BlockingTimeSeriesSplit:
    """
    n_splits: number of folds in cross validation
    """
    def __init__(self, n_splits):
        self.n_splits = n_splits

    def get_n_splits(self, X, y, groups):
        return self.n_splits

    def split(self, X, y=None, groups=None):
        n_samples = len(X)
        k_fold_size = n_samples // self.n_splits
        indices = np.arange(n_samples)

        # margin creates delay between training and validation. Maybe interesting
        # for predictions three/ seven days in advance
        margin = 0

        for i in range(self.n_splits):
            start = i * k_fold_size
            stop = start + k_fold_size
            mid = int(0.8 * (stop - start)) + start
            yield indices[start: mid], indices[mid + margin: stop]


class BlockingTimeSeriesSplitLCPS(BlockingTimeSeriesSplit):
    # return the indices of the time series split
    def return_split(self, X, y=None, groups=None):
        n_samples = len(X)
        k_fold_size = n_samples // self.n_splits
        indices = np.arange(n_samples)

        output = []
        # margin creates delay between training and validation. Maybe interesting
        # for predictions three/ seven days in advance
        margin = 0

        for i in range(self.n_splits):
            start = i * k_fold_size
            stop = start + k_fold_size
            mid = int(0.8 * (stop - start)) + start
            dict_indices = {"train": [start, mid], "validation": [mid + margin, stop]}

            output.append(dict_indices)

        return output


# Below I made an own class to do a grid search to find the best value for
# the hyperparamter lambda (hier alpha)
class GridSearchOwn:
    """
    Class to perform grid search cross validation
    """
    def __init__(self, grid, cv, X, y, model):
        """Initialize attributes"""
        self.grid = grid
        self.cv = cv
        self.X = X
        self.y = y
        
        # create dictionary with model type to call later
        self.model_dict = {"model1": model}

    def perform_search(self):
        """For every value in grid compute the evaluation metric"""
        self.score = dict()
        for par in self.grid:
            model = self.model_dict["model1"](alpha=par)
            self.score["{}".format(par)] = np.mean(cross_val_score(model, self.X, self.y,
                        scoring='neg_mean_absolute_error', cv=self.cv, n_jobs=-1))

        # save best value for the hyper parameter corresponding to the least negative
        # value of the evaluation metric
        self.best_param = max(self.score, key=self.score.get)

# Function to calculate performance metrics
def perf_metrics(y_true, y_pred):
    rsquared = 1 - sum((np.exp(y_true) - np.exp(y_pred))**2)/sum((np.exp(y_true) - np.exp(y_true).mean())**2)
    me = (np.exp(y_true) - np.exp(y_pred)).mean()
    rmse = metrics.mean_squared_error(np.exp(y_true), np.exp(y_pred), squared=False)
    mae = metrics.mean_absolute_error(np.exp(y_true), np.exp(y_pred))
    mape = metrics.mean_absolute_percentage_error(np.exp(y_true), np.exp(y_pred))
    wape = sum(abs(np.exp(y_true) - np.exp(y_pred))) / sum(np.exp(y_true))

    return([rsquared, me, rmse, mae, mape, wape])

# Define
def cross_val_LCPS(y, w, splits_ind):
    for dict in splits_ind:
        return None
