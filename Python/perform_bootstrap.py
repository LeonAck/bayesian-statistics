"""
File to perform block_bootstrap
"""

# Import modules
import pandas as pd
import numpy as np
from cross_validation import BlockingTimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge, Lasso, PoissonRegressor, ElasticNet
from bootstrap_classes import BlockBootstrapStill, BlockBootstrapMoving

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

####### Define necessary objects for bootstrap functions

# define grid for hyperparameter search
grid = dict()
grid['alpha'] = np.arange(0.01, 1, 0.01)

# Define model evaluation method with time series. We use 5 groups
# This is a way of dividing the training set in different validations set,
# while considering the dependence between observations
# Code is found in cross_validation.py
btscv = BlockingTimeSeriesSplit(n_splits=5)


########### Perform bootstrap

# Create class of bootstrap still
bbstill = BlockBootstrapStill(n_blocks=10)

# Perform the bootstrap with Ridge model
# You should be able to insert other models than Ridge. But there seems to be
# some convergence error in Lasso and ElasticNet
bbstill.perform_bootstrap(X, y, Ridge, grid, btscv, scaler)

# print metrics
print("\nmean squared error: {0} "
      "\n mean average error: {1} "
      "\n mean average percentage error: {2}".format(
    bbstill.metrics[0], bbstill.metrics[1], bbstill.metrics[2]))

# Perform bootstrap moving.
# default values of parameters are found in bootstrap_classes.py
bbmoving=BlockBootstrapMoving()
bbmoving.perform_bootstrap(X, y, Ridge, grid, btscv, scaler)

print("\nmean squared error: {0} "
      "\n mean average error: {1} "
      "\n mean average percentage error: {2}".format(
    bbmoving.metrics[0], bbmoving.metrics[1], bbmoving.metrics[2]))
