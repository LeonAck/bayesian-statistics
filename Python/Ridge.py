"""
Perform a Ridge regression
Op basis van deze link https://machinelearningmastery.com/ridge-regression-with-python/
"""

# Import modules
import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedKFold
from sklearn.linear_model import Ridge, Lasso, PoissonRegressor, ElasticNet
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from cross_validation import BlockingTimeSeriesSplit, GridSearchOwn

# Load data
master = pd.read_csv("Data/master.csv", index_col=0)

################### Data preparation ###########################
# mogelijk een deel hiervan naar master index

# Remove all data up to and including 29-10-2020 as variables contain nan
# This is a temporary solution. We might later decide to exclude these variables
# to use the observations before 29-10-2020 as well
master = master.loc['2021-04-19':'2020-10-30']

# replace date where vaccin columns are all zero with nan
master.loc[['2021-04-11'], ['Vacc_Est', 'Vacc_Est_Carehomes', 'Vacc_Adm_GGD',
            'Vacc_Adm_Hosp', 'Vacc_Adm_Doctors']] = np.NAN

# interpolate nans in RNA data and in vaccin columns
master = master.interpolate()

# Create subset of all data with relevant variables.
"""rel_vars = ['ICU_Inflow', 'Hosp_Inflow',
            'Total_Inflow',
            'Tested', 'Cases', 'Cases_Pct',
            'Cases_0_9', 'Cases_10_19', 'Cases_20_29', 'Cases_30_39',
            'Cases_40_49', 'Cases_50_59', 'Cases_60_69', 'Cases_70_79',
            'Cases_80_89', 'Cases_90_Plus', 'Prev_LB', 'Prev', 'Prev_UB',
            'Prev_Growth', 'R',
            'RNA',
            'Vacc_Est']
            """
# master = master[rel_vars]

# Turn data in to numpy ndarray
data = master.values

# We first standardize the data. Otherwise, different features have different
# penalties
# Create object of StandardScaler class
scaler = StandardScaler()

# Perform standardization
data = scaler.fit_transform(data)

# Create X and y data objects. y is ICU inflow
# Remove ICU column to create X
X = np.delete(data, 0, axis=1)

# Keep ICU column for y
y = data[:, 0]

# Split data set into testing and training set. 80% in training set (arbitrary
# choice)
X_train = X[:int(X.shape[0]*0.8)]
X_test = X[int(X.shape[0]*0.8):]
y_train = y[:int(X.shape[0]*0.8)]
y_test = y[int(X.shape[0]*0.8):]

#####################  Create model #######################

# define grid for hyperparameter search
grid = dict()
grid['alpha'] = np.arange(0.01, 1, 0.01)

# Define model evaluation method with time series. We use 5 groups
# This is a way of dividing the training set in different validations set,
# while considering the dependence between observations
# Code is found in cross_validation.py
btscv = BlockingTimeSeriesSplit(n_splits=5)

### Model 1. Programmed GridSearch for hyperparamter my self

######## own
own_grid = GridSearchOwn(grid=grid['alpha'], cv=btscv, X=X_train, y=y_train,
                         model=Ridge)
own_grid.perform_search()

best_lambda = own_grid.best_param
print(best_lambda)
# Define Ridge model with hyper parameter
model = Ridge(alpha=best_lambda)

# Fit the Ridge model on training set
model.fit(X_train, y_train)

# predict y values using the test set
yhat = model.predict(X_test)

# de-standardize the predictions
yhat = yhat * scaler.scale_[0] + scaler.mean_[0]

# add code here to compare predicted results to y_test

### Model 2 GridSearchCV from scikit.learn ###
model2 = Ridge()

# define search
search = GridSearchCV(model2, grid, scoring='neg_mean_absolute_error', cv=btscv,
                      n_jobs=-1)

# perform the search
results = search.fit(X_train, y_train)

# print best parameter and score
print('Best parameter for lambda is: {}'.format(results.best_params_))
print('Best score for this model: {}'.format(np.absolute(results.best_score_)))

# predict values of y based on y_test
yhat2 = search.predict(X_test)

print("standardized predictions", yhat2)

