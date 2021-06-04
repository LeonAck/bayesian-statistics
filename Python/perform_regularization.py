"""
This file contains the following:
    - Standardize data and split into train/test set
    - Compare cross-validation methods and perform
    cross-validation for Ridge, LASSO and Elastic Net
    regression
    - Create Ridge, LASSO and Elastic Net models
    with optimized hyperparameter
    - Calculate one-day ahead predictions for
    SMA(1), SMA(3), SMA(7), Ridge, LASSO, Elastic
    Net models
    - Calculate performance statistics for all models
    above and for the LCPS model

Returns:
    - Results of two cross-validation methods
    - Ridge, LASSO and Elastic Net optimal
    hyperparameter and coefficient estimates
    - Train and test fit performance metrics
    together with Latex output
    - Graphs of predictions and train fit
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


##### Load data #####
data = pd.read_csv("Data/data.csv", index_col=0)


##### Create y, X and train/test split #####

# Create X and y data objects. y is ICU inflow

# Remove ICU column to create X
X = np.delete(data.values, 0, axis=1)

# Keep ICU column for y
y = data.values[:, 0]

# We standardize the regressors. This is required for the regularization models.
# Create object of StandardScaler class
scaler = StandardScaler()

# Standardize X
X = scaler.fit_transform(X)

# Split data into train and test set.
# We use an arbitrary split percentage of 80% in training set
split_pct = 0.8     # Split percentage
split_date = data.index[int(X.shape[0] * split_pct)]    # Split date
X_train = X[:int(X.shape[0] * split_pct)]
X_test = X[int(X.shape[0] * split_pct):]
y_train = y[:int(X.shape[0] * split_pct)]
y_test = y[int(X.shape[0] * split_pct):]
n_train = len(y_train)      # Number of days in train set
n_test = len(y_test)        # Number of days in test set


#####  Hyperparameter Search #####

# Define grid for hyperparameter search
grid = dict()
grid['alpha'] = np.arange(0.01, 100, 0.1)

# Note that lambda and alpha are confusingly used as names for the same parameter.
# This is because the Ridge() function only accepts a parameter alpha.

# Define model evaluation method with time series
# We use cross-validation with 5 splits
# Code is found in cross_validation.py
btscv = BlockingTimeSeriesSplit(n_splits=5)


### Compare self-programmed grid search to GridSearchCV from sklearn

# Own grid search. Code is found in cross_validation.py
own_grid = GridSearchOwn(grid=grid['alpha'], cv=btscv, X=X_train, y=y_train,
                         model=Ridge)
own_grid.perform_search()

# Obtain best lambda
lambda_own = own_grid.best_param
print("Lambda from own grid search:", lambda_own)

# Grid search from sklearn
search = GridSearchCV(Ridge(), grid, scoring='neg_mean_absolute_error',
                      cv=btscv, n_jobs=-1, return_train_score=True)
search.fit(X_train, y_train)

# Obtain best lambda
lambda_sklearn = search.best_params_['alpha']
print("Lambda from sklearn grid search:", lambda_sklearn)

# Note the two lambdas should be equal


##### Regularization Models #####

### Define Ridge model

# Hyperparameter search by means of cross-validation
grid['alpha'] = np.arange(3, 30, 0.01)
search_ridge = GridSearchCV(Ridge(), grid, scoring='neg_mean_absolute_error',
                            cv=btscv, n_jobs=-1, return_train_score=True)
search_ridge.fit(X_train, y_train)

# Save cross-validation score results
lambda_ridge_scores = search_ridge.cv_results_['mean_test_score']

# Graph with cross-validation results
plt.plot(grid['alpha'], lambda_ridge_scores,
         linewidth=3, label='Ridge Cross-Validation')
plt.xlabel('Alpha')
plt.ylabel('Score')
plt.legend()
plt.show()

# Print optimal hyperparameter
print("Ridge lambda: \n", search_ridge.best_params_['alpha'])

# Define Ridge model with optimal hyperparameter
model_ridge = Ridge(alpha=search_ridge.best_params_['alpha'])

# Fit the model on the entire train set
model_ridge.fit(X_train, y_train)

# Print coefficient estimates in ascending order
coef_ridge = pd.DataFrame({'Variable': data.columns[1:],
        'Coefficient': model_ridge.coef_})
print("Ridge Coefficients: \n", coef_ridge.sort_values(by='Coefficient', ascending=False))


### Define Lasso model

# Hyperparameter search by means of cross-validation
grid['alpha'] = np.arange(0.01, 0.15, 0.001)
search_lasso = GridSearchCV(Lasso(), grid, scoring='neg_mean_absolute_error',
                            cv=btscv, n_jobs=-1, return_train_score=True)
search_lasso.fit(X_train, y_train)

# Save cross-validation score results
lambda_lasso_scores = search_lasso.cv_results_['mean_test_score']

# Graph with hyperparameter estimates
plt.plot(grid['alpha'], lambda_lasso_scores,
         linewidth=3, label='Lasso Cross-Validation')
plt.xlabel('Alpha')
plt.ylabel('Score')
plt.legend()
plt.show()

# Print optimal hyperparameter
print("Lasso lambda: \n", search_lasso.best_params_['alpha'])

# Define Lasso model with optimal hyperparameter
model_lasso = Lasso(alpha=search_lasso.best_params_['alpha'])

# Fit the model on train set
model_lasso.fit(X_train, y_train)

# Print positive coefficient estimates in ascending order
coef_lasso = pd.DataFrame({'Variable': data.columns[1:],
        'Coefficient': model_lasso.coef_})
print("Lasso Coefficients: \n", coef_lasso[abs(coef_lasso['Coefficient'])>0].sort_values(by='Coefficient', ascending=False))


### Define Elastic Net model

# Hyperparameter search by means of cross-validation
grid['alpha'] = np.arange(0.01, 0.15, 0.001)
search_elastic = GridSearchCV(ElasticNet(), grid, scoring='neg_mean_absolute_error',
                              cv=btscv, n_jobs=-1, return_train_score=True)
search_elastic.fit(X_train, y_train)

# Save cross-validation score results
lambda_elastic_scores = search_elastic.cv_results_['mean_test_score']

# Graph with hyperparameter estimates
plt.plot(grid['alpha'], lambda_lasso_scores,
         linewidth=3, label='Lasso Cross-Validation')
plt.plot(grid['alpha'], lambda_elastic_scores,
         linewidth=3, label='Elastic Net Cross-Validation')
plt.xlabel('Alpha')
plt.ylabel('Score')
plt.legend()
plt.show()

# Print optimal hyperparameter
print("Elastic Net lambda: \n", search_elastic.best_params_['alpha'])

# Define Elastic Net model with optimal hyperparameter
model_elastic = ElasticNet(alpha=search_elastic.best_params_['alpha'])

# Fit the Lasso model on training set
model_elastic.fit(X_train, y_train)

# Print positive coefficients estimates in ascending order
coef_elastic = pd.DataFrame({'Variable': data.columns[1:],
        'Coefficient': model_elastic.coef_})
print("Elastic Net Coefficients: \n", coef_elastic[abs(coef_elastic['Coefficient'])>0].sort_values(by='Coefficient', ascending=False))


# Dataframe with all coefficients
coef = np.insert(np.array(coef_ridge), 2, coef_lasso['Coefficient'], 1)
coef = np.insert(np.array(coef), 3, coef_elastic['Coefficient'], 1)
coef = pd.DataFrame(coef)


##### Predictions #####

# Define moving average function for numpy arrays
def moving_average(x, w):
    return np.convolve(x, np.ones(w), 'valid') / w

# Create dictionary with one-day predictions on the test set
yhat = {
    'sma1': y[(int(X.shape[0] * split_pct) - 1):(len(y) - 1)],
    'sma3': moving_average(y, 3)[(n_train - 3):(len(moving_average(y, 3)) - 1)],
    'sma7': moving_average(y, 7)[(n_train - 7):(len(moving_average(y, 7)) - 1)],
    'ridge': model_ridge.predict(X_test),
    'lasso': model_lasso.predict(X_test),
    'elastic': model_elastic.predict(X_test),
}

# Load LCPS one-day predictions, computed in perform_LCPS.py
my_file = "y_pred_rolling_LCPS.txt"

with open(my_file, 'r') as f:
    yhat['lcps'] = eval(f.read())


### Performance of predictions

# Performance of train fit
perf_train = pd.DataFrame({'Ridge': perf_metrics(y_train[6:], model_ridge.predict(X_train)[6:]),
        'Lasso': perf_metrics(y_train[6:], model_lasso.predict(X_train)[6:]),
        'Elastic Net': perf_metrics(y_train[6:], model_elastic.predict(X_train)[6:]),
        })
perf_train.index = ['R Squared', 'ME', 'RMSE', 'MAE', 'MAPE', 'WAPE']

# Round to 2 decimals for clarity of the table
perf_train = round(perf_train, 2)

# Print train fit
print("Train fit performance: \n", perf_train)

# Export to latex
print(perf_train.to_latex())


# Performance of test predictions
# We use perf_metrics function in cross_validation.py to calculate performance metrics
perf_test = pd.DataFrame({'SMA(1)':perf_metrics(y_test, yhat['sma1']),
        'SMA(3)':perf_metrics(y_test, yhat['sma3']),
        'SMA(7)':perf_metrics(y_test, yhat['sma7']),
        'LCPS': perf_metrics(y_test, yhat['lcps']),
        'Ridge': perf_metrics(y_test, yhat['ridge']),
        'Lasso': perf_metrics(y_test, yhat['lasso']),
        'Elastic Net': perf_metrics(y_test, yhat['elastic']),
        })
perf_test.index = ['R Squared', 'ME', 'RMSE', 'MAE', 'MAPE', 'WAPE']

# Round to 2 decimals for clarity of the table
perf_test = round(perf_test, 2)

# Print results
print("Test fit performance: \n", perf_test)

# Export to latex
print(perf_test.to_latex())


##### Graphs #####

# Graph of LCPS and SMA(7) predictions
plt.plot(data.index[int(X.shape[0] * split_pct):], np.exp(y_test),
         'k--', linewidth=2, label='ICU Admissions')
plt.plot(data.index[int(X.shape[0] * split_pct):], np.exp(yhat['lcps']),
         '-', linewidth=3, label='LCPS Model')
plt.plot(data.index[int(X.shape[0] * split_pct):], np.exp(yhat['sma7']),
         '-', linewidth=3, label='SMA(7)')
plt.xticks(data.index[np.quantile(range(int(X.shape[0] * split_pct),len(data)), np.linspace(0, 1, 5)).astype(int)])
plt.yticks(np.linspace(35, 70, 8))
plt.xlabel('Time')
plt.ylabel('Admissions')
plt.legend()
plt.show()


# Graph of Ridge, Lasso and Elastic Net predictions
plt.plot(data.index[int(X.shape[0] * split_pct):], np.exp(y_test),
         'k--', linewidth=2, label='ICU Admissions')
plt.plot(data.index[int(X.shape[0] * split_pct):], np.exp(yhat['ridge']),
         '-', linewidth=3, label='Ridge Model')
plt.plot(data.index[int(X.shape[0] * split_pct):], np.exp(yhat['lasso']),
         '-', linewidth=3, label='Lasso Model')
plt.plot(data.index[int(X.shape[0] * split_pct):], np.exp(yhat['elastic']),
         '-', linewidth=3, label='Elastic Net Model')
plt.xticks(data.index[np.quantile(range(int(X.shape[0] * split_pct),len(data)), np.linspace(0, 1, 5)).astype(int)])
plt.yticks(np.linspace(35, 70, 8))
plt.xlabel('Time')
plt.ylabel('Admissions')
plt.legend()
plt.show()


# Graph of train fit of Ridge, Lasso and Elastic Net models
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