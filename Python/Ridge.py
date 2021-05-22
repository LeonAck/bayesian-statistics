"""
Perform a Ridge regression
Op basis van deze link https://machinelearningmastery.com/ridge-regression-with-python/
"""

# Import modules
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedKFold
from sklearn.linear_model import Ridge, Lasso, PoissonRegressor, ElasticNet
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
# (Roel) I don't know why but I need to add the code below
# to be able to import cross_validation
import sys
sys.path.append("Python")
from cross_validation import BlockingTimeSeriesSplit, GridSearchOwn

# Load data
data = pd.read_csv("Data/data.csv", index_col=0)
master = pd.read_csv("Data/master.csv", index_col=0)

# First we standardize the data, since this is required for the Ridge model.
# Create object of StandardScaler class
scaler = StandardScaler()

# Create temporary matrix of values to be standardised,
# excluding the week dummies, ICU inflow
#vars_excl = ['ICU_Inflow', 'Monday', 'Tuesday',
#             'Wednesday', 'Thursday', 'Friday', 'Saturday']
vars_excl = ['ICU_Inflow']
temp = data.loc[:, data.columns.difference(vars_excl)].values

# Standardize temporary matrix
temp = scaler.fit_transform(temp)

# Replace unstandardized values by standardized values
data.loc[:, data.columns.difference(vars_excl)] = temp


# Graph of original Hosp Inflow
plt.plot(master.index, master.Hosp_Inflow, label='Hospital Admissions')
plt.xlabel('Date')
plt.ylabel('Admissions')
plt.legend()
plt.show()

# Graph of log transformed and standardised Hosp Inflow
plt.plot(data.index, data.Hosp_Inflow, label='Hospital Admissions')
plt.xlabel('Date')
plt.ylabel('Admissions')
plt.legend()
plt.show()

# Graph of log transformed and standardised Hosp Inflow
plt.plot(data.index, data.Hosp_Inflow, label='Hospital Admissions')
plt.xlabel('Date')
plt.ylabel('Admissions')
plt.legend()
plt.show()

##### Create y, X and train/test split ################

# Create X and y data objects. y is ICU inflow
# Remove ICU column to create X
X = np.delete(data.values, 0, axis=1)

# Keep ICU column for y
y = data.values[:, 0]

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
# Note this is only necessary when you standardize y
# We do not standardize y since the predictions seem to explode when we do
# yhat = yhat * scaler.scale_[0] + scaler.mean_[0]

# add code here to compare predicted results to y_test

# Graph of predictions
plt.plot(np.linspace(1, len(y_test), len(y_test)), np.exp(y_test), label='ICU Admissions')
plt.plot(np.linspace(1, len(yhat), len(yhat)), np.exp(yhat), label='Predictions')
plt.xlabel('Time')
plt.ylabel('Admissions')
plt.legend()
plt.show()



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

# Graph of predictions
plt.plot(np.linspace(1, len(y_test), len(y_test)), np.exp(y_test), label='ICU Admissions')
plt.plot(np.linspace(1, len(yhat2), len(yhat2)), np.exp(yhat2), label='Predictions')
plt.xlabel('Time')
plt.ylabel('Admissions')
plt.legend()
plt.show()