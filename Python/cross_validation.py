"""
Code from https://hub.packtpub.com/cross-validation-strategies-for-time-series-forecasting-tutorial/
To make BlockingTimeSeriesSplit
"""
import numpy as np
import sklearn.metrics as metrics


class BlockingTimeSeriesSplit:
    """
    Class for Blocking time series split with the API for skicit learn's GridSearchCV function.
    """

    def __init__(self, n_splits):
        self.n_splits = n_splits

    def get_n_splits(self, X, y, groups):
        return self.n_splits

    def split(self, X, y=None, groups=None):
        n_samples = len(X)

        # determine size per fold
        k_fold_size = n_samples // self.n_splits
        indices = np.arange(n_samples)

        # margin creates delay between training and validation. Kept at zero.
        # The validation set immediately follows the training set.
        margin = 0

        for i in range(self.n_splits):
            # determine start index of each fold
            start = i * k_fold_size

            # determine stop index per fold
            stop = start + k_fold_size

            # determine index of boundary between training and validation set in a fold i
            mid = int(0.8 * (stop - start)) + start

            # obtain generator object of the indices
            yield indices[start: mid], indices[mid + margin: stop]


class BlockingTimeSeriesSplitLCPS(BlockingTimeSeriesSplit):
    # return the indices of the time series split
    def return_split(self, X, y=None, groups=None):
        n_samples = len(X)
        k_fold_size = n_samples // self.n_splits

        list_of_dicts_indices = []
        # margin creates delay between training and validation. Maybe interesting
        # for predictions three/ seven days in advance
        margin = 0

        for i in range(self.n_splits):
            # determine start index of each fold
            start = i * k_fold_size

            # determine stop index per fold
            stop = start + k_fold_size

            # determine index of boundary between training and validation set in a fold i
            mid = int(0.8 * (stop - start)) + start

            # save indices in dictionary
            dict_indices = {"train": [start, mid], "validation": [mid + margin, stop]}

            # add dictionary to list
            list_of_dicts_indices.append(dict_indices)

        return list_of_dicts_indices


# Function to calculate performance metrics
def perf_metrics(y_true, y_pred):
    rsquared = 1 - sum((np.exp(y_true) - np.exp(y_pred))**2)/sum((np.exp(y_true) - np.exp(y_true).mean())**2)
    me = (np.exp(y_true) - np.exp(y_pred)).mean()
    rmse = metrics.mean_squared_error(np.exp(y_true), np.exp(y_pred), squared=False)
    mae = metrics.mean_absolute_error(np.exp(y_true), np.exp(y_pred))
    mape = metrics.mean_absolute_percentage_error(np.exp(y_true), np.exp(y_pred))
    wape = sum(abs(np.exp(y_true) - np.exp(y_pred))) / sum(np.exp(y_true))

    return([rsquared, me, rmse, mae, mape, wape])

