"""
Code from https://hub.packtpub.com/cross-validation-strategies-for-time-series-forecasting-tutorial/
To make BlockingTimeSeriesSplit
"""
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_val_score


class BlockingTimeSeriesSplit():
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

        # save best value for the hyper paramter corresponding to the least negative
        # value of the evaluation metric
        self.best_param = max(self.score, key=self.score.get)