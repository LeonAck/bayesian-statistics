"""
Prepare the master dataframe for the models we intend to use,
i.e. using subset of variables, log-transformations, use of lags, etc.

End result is data.csv with unstandardized Y variable as first
column and X variables in subsequent columns
"""

# Import modules
import pandas as pd
import numpy as np


# Load data
master = pd.read_csv("Data/master.csv", index_col=0)

# Create dataframe with variables to be used in actual models
rel_vars = ['ICU_Inflow', 'ICU_Inflow_SMA3d', 'ICU_Inflow_SMA7d',
            'Hosp_Inflow', 'Hosp_Inflow_SMA3d', 'Hosp_Inflow_SMA7d',
            'Cases', 'Cases_SMA3d', 'Cases_SMA7d',
            'Cases_Pct', 'Cases_Pct_SMA3d', 'Cases_Pct_SMA7d',
            'RNA', 'RNA_SMA3d', 'RNA_SMA7d',
            'Vacc_Est', 'Vacc_Est_SMA3d', 'Vacc_Est_SMA7d',
            'Monday', 'Tuesday', 'Wednesday',
            'Thursday', 'Friday', 'Saturday']
data = master.copy()
data = data[rel_vars]


# Take logarithm of variables except for the following
vars_excl = ['Monday', 'Tuesday', 'Wednesday',
             'Thursday', 'Friday', 'Saturday']

# Take logarithm
data.loc[:, data.columns.difference(vars_excl)] = np.log(
    data.loc[:, data.columns.difference(vars_excl)])

# Replace -Inf by 0
data[data == -np.inf] = 0

# Lag all regressors 1 timeperiod since data is only available 1 day after measurement
vars_excl = ['ICU_Inflow',
             'Monday', 'Tuesday', 'Wednesday',
             'Thursday', 'Friday', 'Saturday']
data.loc[:, data.columns.difference(vars_excl)] = data.loc[:, data.columns.difference(vars_excl)].shift(1)

# Insert lagged series of ICU Inflow
data.insert(1, 'ICU_Inflow_Lag', data.ICU_Inflow.shift(1))

# Delete oldest day due to nan's
data.drop(data.head(1).index, inplace=True)

# Save dataframe to file data.csv
data.to_csv(r'Data\data.csv')
