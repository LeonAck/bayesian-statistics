#### first, we load Python modules
import pandas as pd
import numpy as np
import sklearn as skl
# from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import ElasticNet
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge
from sklearn.linear_model import PoissonRegressor
from cross_validation import BlockingTimeSeriesSplit 
from cross_validation import GridSearchOwn

#### second, we load the data and create a copy dataset
master = pd.read_csv("Data/master.csv") # load the data
master2 = master # create a copy dataset

#### third, we transform the data
master = master.set_index("date") # set the date as the index of the data set
master = master.loc['2021-04-19':'2020-10-30'] # remove data that have zeroes as entries
master.loc[['2021-04-11'], ['Vacc_Est', 'Vacc_Est_Carehomes', 'Vacc_Adm_GGD',
            'Vacc_Adm_Hosp', 'Vacc_Adm_Doctors']] = np.NAN # replace remaining zeroes by NAN
master = master.interpolate() # interpolate the missing data

rel_vars = ['ICU_Inflow', 'Hosp_Inflow',
            'Total_Inflow',
            'Tested', 'Cases', 'Cases_Pct',
            'Cases_0_9', 'Cases_10_19', 'Cases_20_29', 'Cases_30_39',
            'Cases_40_49', 'Cases_50_59', 'Cases_60_69', 'Cases_70_79',
            'Cases_80_89', 'Cases_90_Plus', 'Prev_LB', 'Prev', 'Prev_UB',
            'Prev_Growth', 'R',
            'RNA',
            'Vacc_Est']
master = master[rel_vars] # construction of data set that only contains relevant variables
data = master.values # turn data into ndarray

scaler = StandardScaler() # choose the way of standardisation/normalisation
scaler = scaler.fit(data)
normalized = scaler.transform(data) # are these two lines the same as data = scaler.fit_transform(data)?

X = np.delete(data, 0, axis=1)
y = data[:, 0]

#### fourth, we divide the data into train and test sets, currently 80-20 division
X_train = X[:int(X.shape[0]*0.8)]
X_test = X[int(X.shape[0]*0.8):]
y_train = y[:int(X.shape[0]*0.8)]
y_test = y[int(X.shape[0]*0.8):]

#### fifth, we use the sklearn package to model the data
# (1) searching the right value of alpha (lambda)
grid = dict()
grid['alpha'] = np.arange(0.01, 1, 0.01) # we define the grid for hyperparameter search
btscv = BlockingTimeSeriesSplit(n_splits=5) # divide training set in accordance with time series structure (code in cross_validation.py)

# (2) model the elastic net
own_grid1 = GridSearchOwn(grid=grid['alpha'], cv=btscv, X=X_train, y=y_train, model=ElasticNet)
own_grid1.perform_search()
best_lambda1 = own_grid1.best_param
print(best_lambda1)

regr1 = ElasticNet(alpha=best_lambda1) # elastic net model
regr1.fit(X_train,y_train) # compute parameters based on train set
yhat = regr1.predict(X_test) # compute predictions based on test set
yhat = yhat * scaler.scale_[0] + scaler.mean_[0] # de-standardize results

# (3) model the lasso
own_grid2 = GridSearchOwn(grid=grid['alpha'], cv=btscv, X=X_train, y=y_train, model=Lasso)
own_grid2.perform_search()
best_lambda2 = own_grid2.best_param
print(best_lambda2)

regr2 = Lasso(alpha=best_lambda2) # lasso model
regr2.fit(X_train,y_train) # compute parameters based on train set
y2hat = regr2.predict(X_test) # compute predictions based on test set
y2hat = y2hat * scaler.scale_[0] + scaler.mean_[0] # de-standardize results

# (4) model the ridge
own_grid3 = GridSearchOwn(grid=grid['alpha'], cv=btscv, X=X_train, y=y_train, model=Ridge)
own_grid3.perform_search()
best_lambda3 = own_grid3.best_param
print(best_lambda3)

regr3 = Ridge(alpha=best_lambda3) # ridge model
regr3.fit(X_train,y_train) # compute parameters based on test set
y3hat = regr3.predict(X_test) # compute predictions based on train set
y3hat = y3hat * scaler.scale_[0] + scaler.mean_[0] # de-standardize results

# (5) model the poisson regressor (NB poisson regression depends on assumption mean = variance, so can maybe not be used. in lecture it was stressed however that in prediction those assumptions are less important)
own_grid4 = GridSearchOwn(grid=grid['alpha'], cv=btscv, X=X_train, y=y_train, model=PoissonRegressor)
own_grid4.perform_search()
best_lambda4 = own_grid4.best_param
print(best_lambda4)

regr4 = PoissonRegressor(alpha=best_lambda4) # ridge model
regr4.fit(X_train,y_train) # compute parameters based on test set
y4hat = regr4.predict(X_test) # compute predictions based on train set
y4hat = y4hat * scaler.scale_[0] + scaler.mean_[0] # de-standardize results

