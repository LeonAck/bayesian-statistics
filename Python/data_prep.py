"""
Prepare the master dataframe for the models we intend to use,
i.e. using subset of variables, log-transformations, use of lags, etc.

End result is data.csv with unstandardized Y variable as first
column and X variables in subsequent columns
"""

# Import modules
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler


# Load data
data = pd.read_csv("Data/master.csv", index_col=0)


# Take logarithm of variables except for the following
vars_excl = ['Monday', 'Tuesday', 'Wednesday',
             'Thursday', 'Friday', 'Saturday']

# Take logarithm
data.loc[:, data.columns.difference(vars_excl)] = np.log(
    data.loc[:, data.columns.difference(vars_excl)])

# Replace -Inf by 0
data[data == -np.inf] = 0

# Lag all regressors 1 timeperiod since data is only available 1 day after measurement
vars_excl = ['ICU_Inflow', 'Monday', 'Tuesday', 'Wednesday',
             'Thursday', 'Friday', 'Saturday']
data.loc[:, data.columns.difference(vars_excl)] = data.loc[:, data.columns.difference(vars_excl)].shift(-1)

# Add lagged ICU_Inflow variable
data.insert(1, 'ICU_Inflow_Lag1', data.ICU_Inflow.shift(-1))

# Delete oldest day due to nan's
data.drop(data.tail(1).index, inplace=True)




# Save dataframe to file data.csv
data.to_csv(r'Data\data.csv')