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
import sklearn.metrics as metrics
# (Roel) I don't know why but I need to add the code below
# to be able to import cross_validation
import sys
sys.path.append("Python")
from cross_validation import BlockingTimeSeriesSplit, GridSearchOwn


# Load data
master = pd.read_csv("Data/master.csv", index_col=0) #Regressors not lagged (=unrealistic)
data = pd.read_csv("Data/data.csv", index_col=0) #Regressors lagged 1 timeperiod



##### Create y, X and train/test split ################

# Create X and y data objects. y is ICU inflow

# Remove ICU column to create X
X = np.delete(data.values, 0, axis=1)

# Keep ICU column for y
y = data.values[:, 0]

# Split data set into testing and training set. 80% in training set (arbitrary
# choice)
split_date = data.index[int(X.shape[0]*0.8)]
X_train = X[:int(X.shape[0]*0.8)]
X_test = X[int(X.shape[0]*0.8):]
y_train = y[:int(X.shape[0]*0.8)]
y_test = y[int(X.shape[0]*0.8):]


##### Standardization ################

# We standardize the data, since this is required for the Ridge model.
# Create object of StandardScaler class
scaler = StandardScaler()

# Standardize regressor matrix
X_train = scaler.fit_transform(X_train)

# Standardize test regressors using train scaling
X_test = scaler.transform(X_test)

""" Some Graphs
# Graph of ICU Inflow
plt.plot(data.index, data.ICU_Inflow, label='ICU Admissions')
plt.xlabel('Date')
plt.ylabel('Admissions')
plt.legend()
plt.show()

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
"""


#####################  Create model #######################

# Define grid for hyperparameter search
grid = dict()
grid['alpha'] = np.arange(0.01, 1.01, 0.01)

# Define model evaluation method with time series. We use 5 groups
# This is a way of dividing the training set in different validations set,
# while considering the dependence between observations
# Code is found in cross_validation.py
btscv = BlockingTimeSeriesSplit(n_splits=5)


### Compare Grid Searches

######## Own grid search
own_grid = GridSearchOwn(grid=grid['alpha'], cv=btscv, X=X_train, y=y_train,
                         model=Ridge)
own_grid.perform_search()
lambda_own = own_grid.best_param
print(lambda_own)

######## Grid search from sklearn
search = GridSearchCV(Ridge(), grid, scoring='neg_mean_absolute_error',
                      cv=btscv, n_jobs=-1)
search.fit(X_train, y_train)
lambda_sklearn = search.best_params_['alpha']
print(lambda_sklearn) #Note the two lambdas should be equal


#### Define Ridge model
search_ridge = GridSearchCV(Ridge(), grid, scoring='neg_mean_absolute_error',
                            cv=btscv, n_jobs=-1)
search_ridge.fit(X_train, y_train)
print(search_ridge.best_params_['alpha'])

# Define Ridge model with hyper parameter
model_ridge = Ridge(alpha=search_ridge.best_params_['alpha'])

# Fit the Ridge model on training set
model_ridge.fit(X_train, y_train)

# Coefficients
coef_ridge = pd.DataFrame({'Variable': data.columns[1:],
        'Coefficient': model_ridge.coef_})
print(coef_ridge)


#### Define Lasso model
search_lasso = GridSearchCV(Lasso(), grid, scoring='neg_mean_absolute_error',
                            cv=btscv, n_jobs=-1)
search_lasso.fit(X_train, y_train)
print(search_lasso.best_params_['alpha'])

# Define Lasso model with hyper parameter
model_lasso = Lasso(alpha=search_lasso.best_params_['alpha'])

# Fit the Lasso model on training set
model_lasso.fit(X_train, y_train)

# Coefficients
coef_lasso = pd.DataFrame({'Variable': data.columns[1:],
        'Coefficient': model_lasso.coef_})
print(coef_lasso)


#### Define Elastic Net model
search_elastic = GridSearchCV(ElasticNet(), grid, scoring='neg_mean_absolute_error',
                            cv=btscv, n_jobs=-1)
search_elastic.fit(X_train, y_train)
print(search_elastic.best_params_['alpha'])

# Define Elastic Net model with hyper parameter
model_elastic = ElasticNet(alpha=search_elastic.best_params_['alpha'])

# Fit the Lasso model on training set
model_elastic.fit(X_train, y_train)

# Coefficients
coef_elastic = pd.DataFrame({'Variable': data.columns[1:],
        'Coefficient': model_elastic.coef_})
print(coef_elastic)


### Predictions

# Predict y values 1 day ahead using the test set
yhat_ridge = model_ridge.predict(X_test)
yhat_lasso = model_lasso.predict(X_test)
yhat_elastic = model_elastic.predict(X_test)
print(np.exp(yhat_ridge))
print(np.exp(yhat_lasso))
print(np.exp(yhat_elastic))

# Predict y values h day ahead
# Insert code


### Performance of predictions

# Function to calculate performance metrics
def perf_metrics(y_true, y_pred):
    rsquared = 1 - sum((np.exp(y_true) - np.exp(y_pred))**2)/sum((np.exp(y_true) - np.exp(y_true).mean())**2)
    rmse = metrics.mean_squared_error(np.exp(y_true), np.exp(y_pred), squared=False)
    mae = metrics.mean_absolute_error(np.exp(y_true), np.exp(y_pred))
    mape = metrics.mean_absolute_percentage_error(np.exp(y_true), np.exp(y_pred))
    wape = sum(abs(np.exp(y_true) - np.exp(y_pred))) / sum(np.exp(y_true))
    return([rsquared, rmse, mae, mape, wape])

# Compare predicted results to y_test
perf = {'Ridge': perf_metrics(y_test, yhat_ridge),
        'Lasso': perf_metrics(y_test, yhat_lasso),
        'Elastic Net': perf_metrics(y_test, yhat_elastic),
        }
perf = pd.DataFrame(perf)
perf.index = ['R Squared', 'RMSE', 'MAE', 'MAPE', 'WAPE']
print(perf)

# Graph of predictions
plt.plot(data.index[int(X.shape[0]*0.8):], np.exp(y_test), label='ICU Admissions')
plt.plot(data.index[int(X.shape[0]*0.8):], np.exp(yhat_ridge), label='Ridge')
plt.plot(data.index[int(X.shape[0]*0.8):], np.exp(yhat_lasso), label='Lasso')
plt.plot(data.index[int(X.shape[0]*0.8):], np.exp(yhat_elastic), label='Elastic Net')
plt.xlabel('Time')
plt.ylabel('Admissions')
plt.legend()
plt.show()

# Graph of train fit and predictions
plt.plot(data.index, np.exp(y), label='ICU Admissions')
plt.plot(data.index[:int(X.shape[0]*0.8)], np.exp(model_ridge.predict(X_train)), label='Ridge Train Fit')
plt.plot(data.index[:int(X.shape[0]*0.8)], np.exp(model_lasso.predict(X_train)), label='Lasso Train Fit')
plt.plot(data.index[:int(X.shape[0]*0.8)], np.exp(model_elastic.predict(X_train)), label='Elastic Net Train Fit')
plt.plot(data.index[int(X.shape[0]*0.8):], np.exp(yhat_ridge), label='Ridge Test Fit')
plt.plot(data.index[int(X.shape[0]*0.8):], np.exp(yhat_lasso), label='Lasso Test Fit')
plt.plot(data.index[int(X.shape[0]*0.8):], np.exp(yhat_elastic), label='Elastic Net Test Fit')
plt.xlabel('Time')
plt.ylabel('Admissions')
plt.legend()
plt.show()