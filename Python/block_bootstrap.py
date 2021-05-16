import numpy as np
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.linear_model import ElasticNet
from sklearn.linear_model import PoissonRegressors
from cross_validation import BlockingTimeSeriesSplit 
from cross_validation import GridSearchOwn

class BlockBootstrapStill():
    """ Compute statistics using bootstrap method. """
    def __init__(self, n_blocks, margin):
        self.n_blocks = n_blocks # define the number of blocks we want to divide the data in
        self.margin = margin # define the split between train and test set

    def ridge_bootstrap(self, X, y):
        n_observations = len(X) # number of observations is equal to number of rows
        block_size = n_observations // self.n_blocks # define the number of obsersations per block
        indices = np.arange(n_observations) # 0:len(X)
        
        mrse = [] # empty objects to store mean root squared error in
        mae = [] # empty objects to store mean absolute error in
        mape = [] # empty objects to store mean absolute percentage error in

        for i in range(self.n_blocks): # for loop is per block
            start = i * block_size # compute first index of block
            stop = start + block_size # compute stop index of block
            mid = int(0.8 * (stop - start)) + start # compute point between train and test set

            X_train = X[indices[start:mid],] # select train data
            X_test = X[indices[mid + self.margin:stop],] # select test data
            y_train = y[indices[start:mid],] # select train data
            y_test = y[indices[mid + self.margin:stop],] # select test data

            own_grid3 = GridSearchOwn(grid=grid['alpha'], cv=btscv, X=X_train, y=y_train, model=Ridge) # select optimal alpha using gridsearch
            own_grid3.perform_search()
            best_lambda3 = own_grid3.best_param
            print(best_lambda3)
            
            regr3 = Ridge(alpha=best_lambda3) # ridge model
            regr3.fit(X_train,y_train) # compute parameters based on test set
            y3hat = regr3.predict(X_test) # compute predictions based on train set
            y3hat = y3hat * scaler.scale_[0] + scaler.mean_[0] # de-standardize results

            # is dit de juiste vergelijking? of moet je y3hat vergelijken met een waarde na de test set?
            mrse.append(sklearn.metrics.mean_squared_error(y_test,y3hat)) # compute mean squared error and store  in vector above
            mae.append(sklearn.metrics.mean_absolute_error(y_test,y3hat)) # compute mean absolute error and store  in vector above
            mape.append(sklearn.metrics.mean_average_percentage_error(y_test,y3hat)) # compute mean average percentage error and store  in vector above
        
        return [mean(mrse),mean(mae),mean(mape)] # compute the average mrse, mae, mape over the blocks

class BlockBootstrapMoving():
    """ Compute statistics using bootstrap method. """
    def __init__(self, margin, block_size, train_size): # margin = 0, block_size = 40, train_size = 0.8 --> had ik bedacht als default
        self.margin = margin # margin zoals bekend in BlockingTimeSeriesSplit
        self.block_size = block_size # grootte van het block
        self.train_size = train_size # deel van het block dat je als train set wilt; (1-train_size) is deel van het block dat test set is
    
    def ridge_bootstrap(self, X, y):
        indices = np.arrange(len(X)) # 0:len(X)
        mrse = [] # empty objects to store mean root squared error in
        mae = [] # empty objects to store mean absolute error in
        mape = [] # empty objects to store mean absolute percentage error in

        for i in range(0,len(X)-(self.block_size + self.margin + 1)): # idee is dat we vanaf de eerste observatie beginnen en dan block_size aan observaties nemen. Vervolgens itereren met stappen van 1 en voor elk block waarden berekenen
            start = i # first index of block is equal to i
            stop = start + self.block_size + self.margin # stop index of block is equal to start point + the block size and possible margin
            mid = int(self.train_size * (stop - start)) + start # compute point between train and test set

            X_train = X[indices[start:mid],] # select train data
            X_test = X[indices[mid + self.margin:stop],] # select test data
            y_train = y[indices[start:mid],] # select train data
            y_test = y[indices[mid + self.margin:stop],] # select test data

            own_grid3 = GridSearchOwn(grid=grid['alpha'], cv=btscv, X=X_train, y=y_train, model=Ridge) # select optimal alpha using gridsearch
            own_grid3.perform_search()
            best_lambda3 = own_grid3.best_param
            print(best_lambda3)
            
            regr3 = Ridge(alpha=best_lambda3) # ridge model
            regr3.fit(X_train,y_train) # compute parameters based on test set
            y3hat = regr3.predict(X_test) # compute predictions based on train set
            y3hat = y3hat * scaler.scale_[0] + scaler.mean_[0] # de-standardize results

            # is dit de juiste vergelijking? of moet je y3hat vergelijken met een waarde na de test set?
            mrse.append(sklearn.metrics.mean_squared_error(y_test,y3hat)) # compute mean squared error and store  in vector above
            mae.append(sklearn.metrics.mean_absolute_error(y_test,y3hat)) # compute mean absolute error and store  in vector above
            mape.append(sklearn.metrics.mean_average_percentage_error(y_test,y3hat)) # compute mean average percentage error and store  in vector above

        return [mean(mrse),mean(mae),mean(mape)] # compute the average mrse, mae, mape over the blocks