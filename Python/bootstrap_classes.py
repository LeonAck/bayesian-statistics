import numpy as np
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.linear_model import ElasticNet
from sklearn.linear_model import PoissonRegressor
from cross_validation import BlockingTimeSeriesSplit
from cross_validation import GridSearchOwn


class BlockBootstrapStill:
    """ Compute statistics using bootstrap method. """

    def __init__(self, n_blocks, margin=0):
        self.n_blocks = n_blocks  # define the number of blocks we want to divide the data in
        self.margin = margin  # define the split between train and test set

        # create empty object to store metrics
        self.metrics = []

    def perform_bootstrap(self, X, y, model, grid, cv, scaler):
        n_observations = len(X)  # number of observations is equal to number of rows
        block_size = n_observations // self.n_blocks  # define the number of observations per block
        indices = np.arange(n_observations)  # 0:len(X)

        mrse = []  # empty objects to store mean root squared error in
        mae = []  # empty objects to store mean absolute error in
        mape = []  # empty objects to store mean absolute percentage error in

        # create dictionary with model type to call later
        model_dict = {"model1": model}

        for i in range(self.n_blocks):  # for loop is per block
            start = i * block_size  # compute first index of block
            stop = start + block_size  # compute stop index of block
            mid = int(0.8 * (stop - start)) + start  # compute point between train and test set

            X_train = X[indices[start:mid], ]  # select train data
            X_test = X[indices[mid + self.margin:stop], ]  # select test data
            y_train = y[indices[start:mid], ]  # select train data
            y_test = y[indices[mid + self.margin:stop], ]  # select test data

            own_grid3 = GridSearchOwn(grid=grid['alpha'], cv=cv, X=X_train,
                                      y=y_train, model=model_dict["model1"])  # select optimal alpha using gridsearch
            own_grid3.perform_search()
            best_lambda3 = own_grid3.best_param
            print(best_lambda3)

            regr3 = model_dict["model1"](alpha=best_lambda3)  # define model
            regr3.fit(X_train, y_train)  # compute parameters based on test set
            y3hat = regr3.predict(X_test)  # compute predictions based on train set
            y3hat = y3hat * scaler.scale_[0] + scaler.mean_[0]  # de-standardize results

            # hieronder is nu de mean squared error en geen mean root squared error. Klopt dit?
            mrse.append(mean_squared_error(
                y_test, y3hat))  # compute mean squared error and store  in vector above
            mae.append(mean_absolute_error(
                y_test, y3hat))  # compute mean absolute error and store  in vector above
            mape.append(mean_absolute_percentage_error(
                y_test, y3hat))  # compute mean average percentage error and store  in vector above

        self.metrics = [np.mean(mrse), np.mean(mae), np.mean(mape)]  # compute the average mrse, mae, mape over the blocks


class BlockBootstrapMoving:
    """ Compute statistics using bootstrap method. """

    def __init__(self, margin=0, block_size=40,
                 train_size=0.8):  # margin = 0, block_size = 40, train_size = 0.8 --> had ik bedacht als default
        self.margin = margin  # margin zoals bekend in BlockingTimeSeriesSplit
        self.block_size = block_size  # grootte van het block
        self.train_size = train_size  # deel van het block dat je als train set wilt; (1-train_size) is deel van het
        # block dat test set is

        # create empty object to store metrics
        self.metrics = []

    def perform_bootstrap(self, X, y, model, grid, cv, scaler):
        indices = np.arange(len(X))  # 0:len(X)
        mrse = []  # empty objects to store mean root squared error in
        mae = []  # empty objects to store mean absolute error in
        mape = []  # empty objects to store mean absolute percentage error in

        # create dictionary with model type to call later
        model_dict = {"model1": model}

        for i in range(0, len(X) - (
                self.block_size + self.margin + 1)):  # idee is dat we vanaf de eerste observatie beginnen en dan block_size aan observaties nemen. Vervolgens itereren met stappen van 1 en voor elk block waarden berekenen
            start = i  # first index of block is equal to i
            stop = start + self.block_size + self.margin  # stop index of block is equal to start point + the block size and possible margin
            mid = int(self.train_size * (stop - start)) + start  # compute point between train and test set

            X_train = X[indices[start:mid], ]  # select train data
            X_test = X[indices[mid + self.margin:stop], ]  # select test data
            y_train = y[indices[start:mid], ]  # select train data
            y_test = y[indices[mid + self.margin:stop], ]  # select test data

            own_grid3 = GridSearchOwn(grid=grid['alpha'], cv=cv, X=X_train, y=y_train,
                                      model=model_dict["model1"])  # select optimal alpha using gridsearch
            own_grid3.perform_search()
            best_lambda3 = own_grid3.best_param
            print(best_lambda3)

            regr3 = model_dict["model1"](alpha=best_lambda3)  # define model
            regr3.fit(X_train, y_train)  # compute parameters based on test set
            y3hat = regr3.predict(X_test)  # compute predictions based on train set
            y3hat = y3hat * scaler.scale_[0] + scaler.mean_[0]  # de-standardize results

            # hieronder is nu de mean squared error en geen mean root squared error. Klopt dit?
            mrse.append(mean_squared_error(y_test, y3hat))  # compute mean squared error and store  in vector above
            mae.append(mean_absolute_error(y_test, y3hat))  # compute mean absolute error and store  in vector above
            mape.append(mean_absolute_percentage_error(y_test, y3hat))  # compute mean average percentage error and store  in vector above

        self.metrics = [np.mean(mrse), np.mean(mae), np.mean(mape)]  # compute the average mrse, mae, mape over the blocks
