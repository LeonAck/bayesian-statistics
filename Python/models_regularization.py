"""
Perform a Ridge regression
Op basis van deze link https://machinelearningmastery.com/ridge-regression-with-python/
"""

# Import modules
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
import sys
sys.path.append("Python") #Add folder with Python code to directory
from cross_validation import BlockingTimeSeriesSplit, GridSearchOwn, perf_metrics


# Load data
data = pd.read_csv("Data/data.csv", index_col=0)

##### Create y, X and train/test split ################

# Create X and y data objects. y is ICU inflow

# Remove ICU column to create X
X = np.delete(data.values, 0, axis=1)

# Keep ICU column for y
y = data.values[:, 0]

# save weekdays for models michiel


# We standardize the data, since this is required for the Ridge model.
# Create object of StandardScaler class
scaler = StandardScaler()

# Standardize regressor matrix
X = scaler.fit_transform(X)

# Split data set into testing and training set. 80% in training set (arbitrary
# choice)
split_pct = 0.8
split_date = data.index[int(X.shape[0] * split_pct)]
X_train = X[:int(X.shape[0] * split_pct)]
X_test = X[int(X.shape[0] * split_pct):]
y_train = y[:int(X.shape[0] * split_pct)]
y_test = y[int(X.shape[0] * split_pct):]
n_train = len(y_train)
n_test = len(y_test)

#Check whether mean and std are roughly equal for all regressors
X_train.mean(axis=0)
X_test.mean(axis=0)
X_train.std(axis=0)
X_test.std(axis=0)

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
grid['alpha'] = np.arange(0.01, 100, 0.1)

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
print("Lambda from own grid search:", lambda_own)

######## Grid search from sklearn
search = GridSearchCV(Ridge(), grid, scoring='neg_mean_absolute_error',
                      cv=btscv, n_jobs=-1, return_train_score=True)
search.fit(X_train, y_train)
lambda_sklearn = search.best_params_['alpha']
print("Lambda from sklearn grid search:", lambda_sklearn) #Note the two lambdas should be equal


#### Define Ridge model
grid['alpha'] = np.arange(0.1, 100, 0.1)
search_ridge = GridSearchCV(Ridge(), grid, scoring='neg_mean_absolute_error',
                            cv=btscv, n_jobs=-1, return_train_score=True)
search_ridge.fit(X_train, y_train)
print("Ridge lambda: \n", search_ridge.best_params_['alpha'])

# Graph with hyperparameter estimates
lambda_ridge_scores = search_ridge.cv_results_['mean_test_score']
plt.plot(grid['alpha'], lambda_ridge_scores,
         linewidth=3, label='Ridge Cross-Validation')
plt.xlabel('Alpha')
plt.ylabel('Score')
plt.rcParams.update({'font.size': 10})
plt.legend()
plt.show()

# Define Ridge model with hyper parameter
model_ridge = Ridge(alpha=search_ridge.best_params_['alpha'])

# Fit the Ridge model on training set
model_ridge.fit(X_train, y_train)

# Coefficients
coef_ridge = pd.DataFrame({'Variable': data.columns[1:],
        'Coefficient': model_ridge.coef_})
print("Ridge Coefficients: \n", coef_ridge.sort_values(by='Coefficient', ascending=False))

#### Define Lasso model
#grid['alpha'] = np.arange(0.002, 0.015, 0.001)
grid['alpha'] = np.arange(0.006, 0.2, 0.001)
search_lasso = GridSearchCV(Lasso(), grid, scoring='neg_mean_absolute_error',
                            cv=btscv, n_jobs=-1, return_train_score=True)
search_lasso.fit(X_train, y_train)
print("Lasso lambda: \n", search_lasso.best_params_['alpha'])

# Graph with hyperparameter estimates
lambda_lasso_scores = search_lasso.cv_results_['mean_test_score']
plt.plot(grid['alpha'], lambda_lasso_scores,
         linewidth=3, label='Lasso Cross-Validation')
plt.xlabel('Alpha')
plt.ylabel('Score')
plt.legend()
plt.show()

# Define Lasso model with hyper parameter
model_lasso = Lasso(alpha=search_lasso.best_params_['alpha'])

# Fit the Lasso model on training set
model_lasso.fit(X_train, y_train)

# Coefficients
coef_lasso = pd.DataFrame({'Variable': data.columns[1:],
        'Coefficient': model_lasso.coef_})
print("Lasso Coefficients: \n", coef_lasso[abs(coef_lasso['Coefficient'])>0].sort_values(by='Coefficient', ascending=False))

#### Define Elastic Net model
#grid['alpha'] = np.arange(0.002, 0.015, 0.001)
grid['alpha'] = np.arange(0.005, 0.3, 0.001)
search_elastic = GridSearchCV(ElasticNet(), grid, scoring='neg_mean_absolute_error',
                              cv=btscv, n_jobs=-1, return_train_score=True)
search_elastic.fit(X_train, y_train)
print("Elastic Net lambda: \n", search_elastic.best_params_['alpha'])

# Graph with hyperparameter estimates
lambda_elastic_scores = search_elastic.cv_results_['mean_test_score']
#plt.plot(grid['alpha'], lambda_lasso_scores,
#         linewidth=3, label='Lasso Cross-Validation')
plt.plot(grid['alpha'], lambda_elastic_scores,
         linewidth=3, label='Elastic Net Cross-Validation')
plt.xlabel('Alpha')
plt.ylabel('Score')
plt.legend()
plt.show()

# Define Elastic Net model with hyper parameter
model_elastic = ElasticNet(alpha=search_elastic.best_params_['alpha'])

# Fit the Lasso model on training set
model_elastic.fit(X_train, y_train)

# Coefficients
coef_elastic = pd.DataFrame({'Variable': data.columns[1:],
        'Coefficient': model_elastic.coef_})
print("Elastic Net Coefficients: \n", coef_elastic[abs(coef_elastic['Coefficient'])>0].sort_values(by='Coefficient', ascending=False))

# All coefficients
coef = np.insert(np.array(coef_ridge), 2, coef_lasso['Coefficient'], 1)
coef = np.insert(np.array(coef), 3, coef_elastic['Coefficient'], 1)
coef = pd.DataFrame(coef)
coef

### Predictions

# Define moving average function for numpy arrays
def moving_average(x, w):
    return np.convolve(x, np.ones(w), 'valid') / w


# Predict y values 1 day ahead using the test set
yhat_sma1 = y[(int(X.shape[0] * split_pct) - 1):(len(y) - 1)]
yhat_sma3 = moving_average(y, 3)[(n_train - 3):(len(moving_average(y, 3)) - 1)]
yhat_sma7 = moving_average(y, 7)[(n_train - 7):(len(moving_average(y, 7)) - 1)]
yhat_ridge = model_ridge.predict(X_test)
yhat_lasso = model_lasso.predict(X_test)
yhat_elastic = model_elastic.predict(X_test)

# Predict y values x day ahead using the test set
vars_excl = list(range(data.columns.get_loc("Monday")-1,data.columns.get_loc("Monday")+5))
vars_incl = np.delete(list(range(X.shape[1] - 1)), vars_excl)
seas = X_test[:,vars_excl].copy()
yhat = {}
for i in range(7):
    # Create temporary array with regressors shifted i periods
    temp_i = X[int(X.shape[0] * split_pct) - i:X.shape[0] - i].copy()
    temp_i[:, vars_excl] = seas.copy()
    X_test_i = temp_i.copy()  #Create X_test for predicting i periods ahead

    # Predict i periods ahead and assign to dictionary key
    yhat['yhat_ridge_' + str(i+1)] = model_ridge.predict(X_test_i)
    yhat['yhat_lasso_' + str(i + 1)] = model_lasso.predict(X_test_i)
    yhat['yhat_elastic_' + str(i + 1)] = model_elastic.predict(X_test_i)

"""
# Predict y values 3 days ahead
yhat3_sma1 = y[(int(X.shape[0]*split_pct)-3):(len(y)-3)]
yhat3_sma3 = np.zeros(len(yhat3_sma1))
for t in range(len(yhat3_sma1)):
    at = y[(int(X.shape[0]*split_pct) - 3 + t)]
    at1 = y[(int(X.shape[0]*split_pct) - 4 + t)]
    at2 = y[(int(X.shape[0]*split_pct) - 5 + t)]
    sma_at = moving_average(np.array([at, at1, at2]), 3)
    sma_at1 = moving_average([sma_at[0], at, at1], 3)
    yhat3_sma3[t] = moving_average([sma_at1[0], sma_at[0], at], 3)[0]
yhat3_sma7 = np.zeros(len(yhat3_sma1))
for t in range(len(yhat3_sma1)):
    at = y[(int(X.shape[0]*split_pct) - 3 + t)]
    at1 = y[(int(X.shape[0]*split_pct) - 4 + t)]
    at2 = y[(int(X.shape[0]*split_pct) - 5 + t)]
    at3 = y[(int(X.shape[0] * split_pct) - 6 + t)]
    at4 = y[(int(X.shape[0] * split_pct) - 7 + t)]
    at5 = y[(int(X.shape[0] * split_pct) - 8 + t)]
    at6 = y[(int(X.shape[0] * split_pct) - 9 + t)]
    sma_at = moving_average(np.array([at, at1, at2, at3, at4, at5, at6]), 7)
    sma_at1 = moving_average([sma_at[0], at, at1, at2, at3, at4, at5], 7)
    sma_at2 = moving_average([sma_at1[0], sma_at[0], at, at1, at2, at3, at4], 7)
    yhat3_sma7[t] = moving_average([sma_at1[0], sma_at[0], at, at1, at2, at3, at4], 7)[0]

# Predict y values 7 days ahead
yhat7_sma1 = y[(int(X.shape[0]*split_pct)-7):(len(y)-7)]
yhat7_sma3 = np.zeros(len(yhat7_sma1))
for t in range(len(yhat7_sma1)):
    at = y[(int(X.shape[0]*split_pct) - 3 + t)]
    at1 = y[(int(X.shape[0]*split_pct) - 4 + t)]
    at2 = y[(int(X.shape[0]*split_pct) - 5 + t)]
    sma_at = moving_average(np.array([at, at1, at2]), 3)
    sma_at1 = moving_average([sma_at[0], at, at1], 3)
    sma_at2 = moving_average([sma_at1[0], sma_at[0], at], 3)
    sma_at3 = moving_average([sma_at2[0], sma_at1[0], sma_at[0]], 3)
    sma_at4 = moving_average([sma_at3[0], sma_at2[0], sma_at1[0]], 3)
    sma_at5 = moving_average([sma_at4[0], sma_at3[0], sma_at2[0]], 3)
    yhat7_sma3[t] = moving_average([sma_at5[0], sma_at4[0], sma_at3[0]], 3)[0]
yhat7_sma7 = np.zeros(len(yhat7_sma1))
for t in range(len(yhat7_sma1)):
    at = y[(int(X.shape[0]*split_pct) - 3 + t)]
    at1 = y[(int(X.shape[0]*split_pct) - 4 + t)]
    at2 = y[(int(X.shape[0]*split_pct) - 5 + t)]
    at3 = y[(int(X.shape[0] * split_pct) - 6 + t)]
    at4 = y[(int(X.shape[0] * split_pct) - 7 + t)]
    at5 = y[(int(X.shape[0] * split_pct) - 8 + t)]
    at6 = y[(int(X.shape[0] * split_pct) - 9 + t)]
    sma_at = moving_average(np.array([at, at1, at2, at3, at4, at5, at6]), 7)
    sma_at1 = moving_average([sma_at[0], at, at1, at2, at3, at4, at5], 7)
    sma_at2 = moving_average([sma_at1[0], sma_at[0], at, at1, at2, at3, at4], 7)
    sma_at3 = moving_average([sma_at2[0], sma_at1[0], sma_at[0], at, at1, at2, at3], 7)
    sma_at4 = moving_average([sma_at3[0], sma_at2[0], sma_at1[0], sma_at[0], at, at1, at2], 7)
    sma_at5 = moving_average([sma_at4[0], sma_at3[0], sma_at2[0], sma_at1[0], sma_at[0], at, at1], 7)
    yhat7_sma7[t] = moving_average([sma_at5[0], sma_at4[0], sma_at3[0], sma_at2[0], sma_at1[0], sma_at[0], at], 7)[0]
"""

# load LCPS_rolling one-day predictions
my_file = "y_pred_rolling_LCPS.txt"

with open(my_file, 'r') as f:
    yhat_lcps_oneday = eval(f.read())

### Performance of predictions

# Performance of train fit
perf_train = {'SMA(1)':perf_metrics(y_train[6:], y_train[5:(len(y_train)-1)]),
        'SMA(3)':perf_metrics(y_train[6:], moving_average(y, 3)[4:(len(y_train)-2)]),
        'SMA(7)':perf_metrics(y_train[6:], moving_average(y, 7)[:(len(y_train)-6)]),
        'Ridge': perf_metrics(y_train[6:], model_ridge.predict(X_train)[6:]),
        'Lasso': perf_metrics(y_train[6:], model_lasso.predict(X_train)[6:]),
        'Elastic Net': perf_metrics(y_train[6:], model_elastic.predict(X_train)[6:]),
        }
perf_train = pd.DataFrame(perf_train)
perf_train = round(perf_train, 2)
perf_train.index = ['R Squared', 'ME', 'RMSE', 'MAE', 'MAPE', 'WAPE']
print("Train fit performance: \n", perf_train)

# Export to latex
print(perf_train.to_latex())

# Performance of test predictions
perf_test = {'SMA(1)':perf_metrics(y_test, yhat_sma1),
        'SMA(3)':perf_metrics(y_test, yhat_sma3),
        'SMA(7)':perf_metrics(y_test, yhat_sma7),
#        'LCPS': perf_metrics(y_test, yhat_lcps_oneday),
        'Ridge': perf_metrics(y_test, yhat_ridge),
        'Lasso': perf_metrics(y_test, yhat_lasso),
        'Elastic Net': perf_metrics(y_test, yhat_elastic),
        }
perf_test = pd.DataFrame(perf_test)
perf_test = round(perf_test, 2)
perf_test.index = ['R Squared', 'ME', 'RMSE', 'MAE', 'MAPE', 'WAPE']
print("Test fit performance: \n", perf_test)

# Export to latex
print(perf_test.to_latex())


# Performance of test predictions for multi day predictions
perf_test_pred = {}
for i in range(7):
    perf_test_pred['Ridge ' + str(i+1)] = perf_metrics(y_test, yhat['yhat_ridge_' + str(i+1)])
    perf_test_pred['Lasso ' + str(i + 1)] = perf_metrics(y_test, yhat['yhat_lasso_' + str(i + 1)])
    perf_test_pred['Elastic Net ' + str(i+1)] = perf_metrics(y_test, yhat['yhat_elastic_' + str(i+1)])
perf_test_pred = pd.DataFrame(perf_test_pred)
perf_test_pred = round(perf_test_pred, 2)
perf_test_pred.index = ['R Squared', 'ME', 'RMSE', 'MAE', 'MAPE', 'WAPE']
print("Test fit performance: \n", perf_test_pred)

keys_ridge = ['Ridge 1', 'Ridge 2', 'Ridge 3', 'Ridge 4', 'Ridge 5', 'Ridge 6', 'Ridge 7']
perf_test_pred[keys_ridge]

perf_test = {'Ridge_1': perf_metrics(y_test, yhat[yhat_ridge_1]),
        'Lasso': perf_metrics(y_test, yhat_lasso),
        'Elastic Net': perf_metrics(y_test, yhat_elastic),
        }
perf_test = pd.DataFrame(perf_test)
perf_test = round(perf_test, 2)
perf_test.index = ['R Squared', 'ME', 'RMSE', 'MAE', 'MAPE', 'WAPE']
print("Test fit performance: \n", perf_test)

"""
# Compare predicted results to y_test for 3 day ahead predictions
perf = {'SMA(1)':perf_metrics(y_test, yhat3_sma1),
        'SMA(3)':perf_metrics(y_test, yhat3_sma3),
        'SMA(7)':perf_metrics(y_test, yhat3_sma7),
        }
perf = pd.DataFrame(perf)
perf.index = ['R Squared', 'RMSE', 'MAE', 'MAPE', 'WAPE']
print(perf)

# Compare predicted results to y_test for 7 day ahead predictions
perf = {'SMA(1)':perf_metrics(y_test, yhat7_sma1),
        'SMA(3)':perf_metrics(y_test, yhat7_sma3),
        'SMA(7)':perf_metrics(y_test, yhat7_sma7),
        }
perf = pd.DataFrame(perf)
perf.index = ['R Squared', 'RMSE', 'MAE', 'MAPE', 'WAPE']
print(perf)
"""


# Graph of predictions
plt.plot(data.index[int(X.shape[0] * split_pct):], np.exp(y_test),
         'k--', linewidth=2, label='ICU Admissions')
#plt.plot(data.index[int(X.shape[0] * split_pct):], np.exp(yhat_lcps_oneday),
#         '-', linewidth=3, label='LCPS Model')
plt.plot(data.index[int(X.shape[0] * split_pct):], np.exp(yhat_sma7),
         '-', linewidth=3, label='SMA(7)')
plt.xticks(data.index[np.quantile(range(int(X.shape[0] * split_pct),len(data)), np.linspace(0, 1, 5)).astype(int)])
plt.yticks(np.linspace(35, 70, 8))
plt.xlabel('Time')
plt.ylabel('Admissions')
plt.legend()
plt.show()

"""
# Graph of predictions
plt.plot(data.index[int(X.shape[0] * split_pct):], np.exp(y_test),
         linewidth=2, label='ICU Admissions')
plt.plot(data.index[int(X.shape[0] * split_pct):], np.exp(yhat_lcps_oneday),
         '--', linewidth=2, label='LCPS Model')
plt.xticks(data.index[np.quantile(range(int(X.shape[0] * split_pct),len(data)), np.linspace(0, 1, 5)).astype(int)])
plt.yticks(np.linspace(35, 70, 8))
plt.xlabel('Time')
plt.ylabel('Admissions')
plt.legend()
plt.show()
"""

# Graph of predictions
plt.plot(data.index[int(X.shape[0] * split_pct):], np.exp(y_test),
         'k--', linewidth=2, label='ICU Admissions')
plt.plot(data.index[int(X.shape[0] * split_pct):], np.exp(yhat_ridge),
         '-', linewidth=3, label='Ridge Model')
plt.plot(data.index[int(X.shape[0] * split_pct):], np.exp(yhat_lasso),
         '-', linewidth=3, label='Lasso Model')
plt.plot(data.index[int(X.shape[0] * split_pct):], np.exp(yhat_elastic),
         '-', linewidth=3, label='Elastic Net Model')
plt.xticks(data.index[np.quantile(range(int(X.shape[0] * split_pct),len(data)), np.linspace(0, 1, 5)).astype(int)])
plt.yticks(np.linspace(35, 70, 8))
plt.xlabel('Time')
plt.ylabel('Admissions')
#plt.rc('font', size=10)
plt.legend()
plt.show()

# Graph of train fit
plt.plot(data.index[6:int(X.shape[0] * split_pct)], np.exp(y_train)[6:], label='ICU Admissions')
#plt.plot(data.index[:int(X.shape[0] * split_pct)], np.exp(yhat_lcps_oneday), label='LCPS Model')
plt.plot(data.index[6:int(X.shape[0] * split_pct)], np.exp(moving_average(y, 3)[4:(len(y_train)-2)]),
         label='SMA(3) Model')
plt.plot(data.index[6:int(X.shape[0] * split_pct)], np.exp(moving_average(y, 7)[:(len(y_train)-6)]),
         label='SMA(7) Model')
plt.xticks(data.index[np.quantile(range(6,int(X.shape[0] * split_pct)), np.linspace(0, 1, 5)).astype(int)])
plt.xlabel('Time')
plt.ylabel('Admissions')
plt.legend()
plt.show()

# Graph of train fit
plt.plot(data.index[:int(X.shape[0] * split_pct)], np.exp(y_train), label='ICU Admissions')
plt.plot(data.index[:int(X.shape[0] * split_pct)], np.exp(model_ridge.predict(X_train)),
         label='Ridge Model')
plt.plot(data.index[:int(X.shape[0] * split_pct)], np.exp(model_lasso.predict(X_train)),
         label='Lasso Model')
plt.plot(data.index[:int(X.shape[0] * split_pct)], np.exp(model_elastic.predict(X_train)),
         label='Elastic Net Model')
plt.xticks(data.index[np.quantile(range(int(X.shape[0] * split_pct)), np.linspace(0, 1, 5)).astype(int)])
plt.xlabel('Time')
plt.ylabel('Admissions')
plt.legend()
plt.show()

# Graph of train fit and predictions
plt.plot(data.index, np.exp(y), label='ICU Admissions')
plt.plot(data.index[:int(X.shape[0] * split_pct)], np.exp(model_elastic.predict(X_train)),
         label='Elastic Net Train Fit')
plt.plot(data.index[int(X.shape[0] * split_pct):], np.exp(yhat_elastic), label='Elastic Net Test Fit')
plt.xticks(data.index[np.quantile(range(0,len(data)), np.linspace(0, 1, 5)).astype(int)])
plt.xlabel('Time')
plt.ylabel('Admissions')
plt.legend()
plt.show()