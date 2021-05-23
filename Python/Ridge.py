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
master = pd.read_csv("Data/master.csv", index_col=0)
data = pd.read_csv("Data/data.csv", index_col=0)
data1 = pd.read_csv("Data/data1.csv", index_col=0) #regressors lagged 1 day
data2 = pd.read_csv("Data/data2.csv", index_col=0) #regressors lagged 2 day
data3 = pd.read_csv("Data/data3.csv", index_col=0)
data4 = pd.read_csv("Data/data4.csv", index_col=0)
data5 = pd.read_csv("Data/data5.csv", index_col=0)
data6 = pd.read_csv("Data/data6.csv", index_col=0)
data7 = pd.read_csv("Data/data7.csv", index_col=0)

# We switch the time order so the earliest date comes first
master = master.sort_index()
data = data.sort_index()
data1 = data1.sort_index()
data2 = data2.sort_index()
data3 = data3.sort_index()
data4 = data4.sort_index()
data5 = data5.sort_index()
data6 = data6.sort_index()
data7 = data7.sort_index()


##### Create y, X and train/test split ################

# Create X and y data objects. y is ICU inflow

# Remove ICU column to create X
X1 = np.delete(data1.values, 0, axis=1)
X2 = np.delete(data2.values, 0, axis=1)
X3 = np.delete(data3.values, 0, axis=1)
X4 = np.delete(data4.values, 0, axis=1)
X5 = np.delete(data5.values, 0, axis=1)
X6 = np.delete(data6.values, 0, axis=1)
X7 = np.delete(data7.values, 0, axis=1)

# Keep ICU column for y
y = data.values[:, 0]

# Split data set into testing and training set. 80% in training set (arbitrary
# choice)
split_date = data.index[int(X1.shape[0]*0.8)]
X1_train = X1[:int(X1.shape[0]*0.8)]
X2_train = X2[:int(X2.shape[0]*0.8)]
X3_train = X3[:int(X3.shape[0]*0.8)]
X4_train = X4[:int(X4.shape[0]*0.8)]
X5_train = X5[:int(X5.shape[0]*0.8)]
X6_train = X6[:int(X6.shape[0]*0.8)]
X7_train = X7[:int(X7.shape[0]*0.8)]
X1_test = X1[int(X1.shape[0]*0.8):]
X2_test = X2[int(X2.shape[0]*0.8):]
X3_test = X3[int(X3.shape[0]*0.8):]
X4_test = X4[int(X4.shape[0]*0.8):]
X5_test = X5[int(X5.shape[0]*0.8):]
X6_test = X6[int(X6.shape[0]*0.8):]
X7_test = X7[int(X7.shape[0]*0.8):]
y_train = y[:int(X1.shape[0]*0.8)]
y_test = y[int(X1.shape[0]*0.8):]


##### Standardization ################

# We standardize the data, since this is required for the Ridge model.
# Create object of StandardScaler class
scaler1 = StandardScaler()
scaler2 = StandardScaler()
scaler3 = StandardScaler()
scaler4 = StandardScaler()
scaler5 = StandardScaler()
scaler6 = StandardScaler()
scaler7 = StandardScaler()

# Standardize regressor matrix
X1_train = scaler1.fit_transform(X1_train)
X2_train = scaler2.fit_transform(X2_train)
X3_train = scaler3.fit_transform(X3_train)
X4_train = scaler4.fit_transform(X4_train)
X5_train = scaler5.fit_transform(X5_train)
X6_train = scaler6.fit_transform(X6_train)
X7_train = scaler7.fit_transform(X7_train)

# Standardize test regressors using train scaling
X1_test = scaler1.transform(X1_test)
X2_test = scaler2.transform(X2_test)
X3_test = scaler3.transform(X3_test)
X4_test = scaler4.transform(X4_test)
X5_test = scaler5.transform(X5_test)
X6_test = scaler6.transform(X6_test)
X7_test = scaler7.transform(X7_test)



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
own_grid1 = GridSearchOwn(grid=grid['alpha'], cv=btscv, X=X1_train, y=y_train,
                         model=Ridge)
own_grid2 = GridSearchOwn(grid=grid['alpha'], cv=btscv, X=X2_train, y=y_train,
                         model=Ridge)
own_grid3 = GridSearchOwn(grid=grid['alpha'], cv=btscv, X=X3_train, y=y_train,
                         model=Ridge)
own_grid4 = GridSearchOwn(grid=grid['alpha'], cv=btscv, X=X4_train, y=y_train,
                         model=Ridge)
own_grid5 = GridSearchOwn(grid=grid['alpha'], cv=btscv, X=X5_train, y=y_train,
                         model=Ridge)
own_grid6 = GridSearchOwn(grid=grid['alpha'], cv=btscv, X=X6_train, y=y_train,
                         model=Ridge)
own_grid7 = GridSearchOwn(grid=grid['alpha'], cv=btscv, X=X7_train, y=y_train,
                         model=Ridge)
own_grid1.perform_search()
own_grid2.perform_search()
own_grid3.perform_search()
own_grid4.perform_search()
own_grid5.perform_search()
own_grid6.perform_search()
own_grid7.perform_search()

best_lambda1 = own_grid1.best_param
best_lambda2 = own_grid2.best_param
best_lambda3 = own_grid3.best_param
best_lambda4 = own_grid4.best_param
best_lambda5 = own_grid5.best_param
best_lambda6 = own_grid6.best_param
best_lambda7 = own_grid7.best_param
print(best_lambda1, best_lambda2, best_lambda3,
      best_lambda4, best_lambda5, best_lambda6,
      best_lambda7,)

# Define Ridge model with hyper parameter
model1 = Ridge(alpha=best_lambda1)
model2 = Ridge(alpha=best_lambda2)
model3 = Ridge(alpha=best_lambda3)
model4 = Ridge(alpha=best_lambda4)
model5 = Ridge(alpha=best_lambda5)
model6 = Ridge(alpha=best_lambda6)
model7 = Ridge(alpha=best_lambda7)

# Fit the Ridge model on training set
model1.fit(X1_train, y_train)
model2.fit(X2_train, y_train)
model3.fit(X3_train, y_train)
model4.fit(X4_train, y_train)
model5.fit(X5_train, y_train)
model6.fit(X6_train, y_train)
model7.fit(X7_train, y_train)

# predict y values using the test set
yhat1 = model1.predict(X1_test)
yhat2 = model2.predict(X2_test)
yhat3 = model3.predict(X3_test)
yhat4 = model4.predict(X4_test)
yhat5 = model5.predict(X5_test)
yhat6 = model6.predict(X6_test)
yhat7 = model7.predict(X7_test)

# de-standardize the predictions
# Note this is only necessary when you standardize y
# We do not standardize y since the predictions seem to explode when we do
# yhat = yhat * scaler.scale_[0] + scaler.mean_[0]

# Coefficients
coef1 = pd.DataFrame({'Variable': data1.columns[1:],
        'Coefficient': model1.coef_})
coef2 = pd.DataFrame({'Variable': data2.columns[1:],
        'Coefficient': model2.coef_})
coef3 = pd.DataFrame({'Variable': data3.columns[1:],
        'Coefficient': model3.coef_})
coef4 = pd.DataFrame({'Variable': data4.columns[1:],
        'Coefficient': model4.coef_})
coef5 = pd.DataFrame({'Variable': data5.columns[1:],
        'Coefficient': model5.coef_})
coef6 = pd.DataFrame({'Variable': data6.columns[1:],
        'Coefficient': model6.coef_})
coef7 = pd.DataFrame({'Variable': data7.columns[1:],
        'Coefficient': model7.coef_})

# Compare predicted results to y_test
perf = {'Ridge1': [metrics.mean_squared_error(np.exp(y_test), np.exp(yhat1), squared=False),
    metrics.mean_absolute_error(np.exp(y_test), np.exp(yhat1)),
    metrics.mean_absolute_percentage_error(np.exp(y_test), np.exp(yhat1)),
    sum(abs(np.exp(y_test) - np.exp(yhat1)))/sum(np.exp(y_test))],
        'Ridge2': [metrics.mean_squared_error(np.exp(y_test), np.exp(yhat2), squared=False),
    metrics.mean_absolute_error(np.exp(y_test), np.exp(yhat2)),
    metrics.mean_absolute_percentage_error(np.exp(y_test), np.exp(yhat2)),
    sum(abs(np.exp(y_test) - np.exp(yhat2)))/sum(np.exp(y_test))],
        'Ridge3': [metrics.mean_squared_error(np.exp(y_test), np.exp(yhat3), squared=False),
                   metrics.mean_absolute_error(np.exp(y_test), np.exp(yhat3)),
                   metrics.mean_absolute_percentage_error(np.exp(y_test), np.exp(yhat3)),
                   sum(abs(np.exp(y_test) - np.exp(yhat3))) / sum(np.exp(y_test))],
        'Ridge4': [metrics.mean_squared_error(np.exp(y_test), np.exp(yhat4), squared=False),
                   metrics.mean_absolute_error(np.exp(y_test), np.exp(yhat4)),
                   metrics.mean_absolute_percentage_error(np.exp(y_test), np.exp(yhat4)),
                   sum(abs(np.exp(y_test) - np.exp(yhat4))) / sum(np.exp(y_test))],
        'Ridge5': [metrics.mean_squared_error(np.exp(y_test), np.exp(yhat5), squared=False),
                   metrics.mean_absolute_error(np.exp(y_test), np.exp(yhat5)),
                   metrics.mean_absolute_percentage_error(np.exp(y_test), np.exp(yhat5)),
                   sum(abs(np.exp(y_test) - np.exp(yhat5))) / sum(np.exp(y_test))],
        'Ridge6': [metrics.mean_squared_error(np.exp(y_test), np.exp(yhat6), squared=False),
                   metrics.mean_absolute_error(np.exp(y_test), np.exp(yhat6)),
                   metrics.mean_absolute_percentage_error(np.exp(y_test), np.exp(yhat6)),
                   sum(abs(np.exp(y_test) - np.exp(yhat6))) / sum(np.exp(y_test))],
        'Ridge7': [metrics.mean_squared_error(np.exp(y_test), np.exp(yhat7), squared=False),
                   metrics.mean_absolute_error(np.exp(y_test), np.exp(yhat7)),
                   metrics.mean_absolute_percentage_error(np.exp(y_test), np.exp(yhat7)),
                   sum(abs(np.exp(y_test) - np.exp(yhat7))) / sum(np.exp(y_test))]
        }
perf = pd.DataFrame(perf)
perf.index = ['RMSE', 'MAE', 'MAPE', 'WAPE']
print(perf)

# Graph of predictions
plt.plot(data1.index[int(X1.shape[0]*0.8):], np.exp(y_test), label='ICU Admissions')
plt.plot(data1.index[int(X1.shape[0]*0.8):], np.exp(yhat1), label='Predictions')
plt.plot(data1.index[int(X1.shape[0]*0.8):], np.exp(yhat2), label='Predictions')
plt.plot(data1.index[int(X1.shape[0]*0.8):], np.exp(yhat3), label='Predictions')
plt.xlabel('Time')
plt.ylabel('Admissions')
plt.legend()
plt.show()

"""
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

print("standardized predictions", np.exp(yhat2))

# Compare predicted results to y_test
perf2 = {'Ridge': [metrics.mean_squared_error(np.exp(y_test), np.exp(yhat2), squared=False),
    metrics.mean_absolute_error(np.exp(y_test), np.exp(yhat2)),
    metrics.mean_absolute_percentage_error(np.exp(y_test), np.exp(yhat2)),
    sum(abs(np.exp(y_test) - np.exp(yhat2)))/sum(np.exp(y_test))]}
perf2 = pd.DataFrame(perf)
perf2.index = ['RMSE', 'MAE', 'MAPE', 'WAPE']

# Graph of predictions
plt.plot(np.linspace(1, len(y_test), len(y_test)), np.exp(y_test), label='ICU Admissions')
plt.plot(np.linspace(1, len(yhat2), len(yhat2)), np.exp(yhat2), label='Predictions')
plt.xlabel('Time')
plt.ylabel('Admissions')
plt.legend()
plt.show()

# Graph of train fit and predictions
plt.plot(data.index, np.exp(y), label='ICU Admissions')
plt.plot(data.index[:int(X.shape[0]*0.8)], np.exp(search.predict(X_train)), label='Train Fit')
plt.plot(data.index[int(X.shape[0]*0.8):], np.exp(yhat2), label='Predictions')
plt.xlabel('Time')
plt.ylabel('Admissions')
plt.legend()
plt.show()
"""