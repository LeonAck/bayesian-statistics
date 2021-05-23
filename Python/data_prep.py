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
master = pd.read_csv("Data/master.csv", index_col=0)

# Create dataframe with variables to be used in actual models
rel_vars = ['ICU_Inflow', 'ICU_Inflow_SMA7d',
            'Hosp_Inflow', 'Hosp_Inflow_SMA7d',
            'Cases', 'Cases_Pct',
            'RNA', 'Vacc_Est',
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
data1 = data.copy()
data2 = data.copy()
data3 = data.copy()
data4 = data.copy()
data5 = data.copy()
data6 = data.copy()
data7 = data.copy()
data1.loc[:, data1.columns.difference(vars_excl)] = data1.loc[:, data1.columns.difference(vars_excl)].shift(-1)
data1.insert(1, 'ICU_Inflow_Lag', data1.ICU_Inflow.shift(-1))
data2.loc[:, data2.columns.difference(vars_excl)] = data2.loc[:, data2.columns.difference(vars_excl)].shift(-2)
data2.insert(1, 'ICU_Inflow_Lag', data2.ICU_Inflow.shift(-2))
data3.loc[:, data3.columns.difference(vars_excl)] = data3.loc[:, data3.columns.difference(vars_excl)].shift(-3)
data3.insert(1, 'ICU_Inflow_Lag', data3.ICU_Inflow.shift(-3))
data4.loc[:, data4.columns.difference(vars_excl)] = data4.loc[:, data4.columns.difference(vars_excl)].shift(-4)
data4.insert(1, 'ICU_Inflow_Lag', data4.ICU_Inflow.shift(-4))
data5.loc[:, data5.columns.difference(vars_excl)] = data5.loc[:, data5.columns.difference(vars_excl)].shift(-5)
data5.insert(1, 'ICU_Inflow_Lag', data5.ICU_Inflow.shift(-5))
data6.loc[:, data6.columns.difference(vars_excl)] = data6.loc[:, data6.columns.difference(vars_excl)].shift(-6)
data6.insert(1, 'ICU_Inflow_Lag', data6.ICU_Inflow.shift(-6))
data7.loc[:, data7.columns.difference(vars_excl)] = data7.loc[:, data7.columns.difference(vars_excl)].shift(-7)
data7.insert(1, 'ICU_Inflow_Lag', data7.ICU_Inflow.shift(-7))


# Delete oldest week due to nan's
data.drop(data.tail(7).index, inplace=True)
data1.drop(data1.tail(7).index, inplace=True)
data2.drop(data2.tail(7).index, inplace=True)
data3.drop(data3.tail(7).index, inplace=True)
data4.drop(data4.tail(7).index, inplace=True)
data5.drop(data5.tail(7).index, inplace=True)
data6.drop(data6.tail(7).index, inplace=True)
data7.drop(data7.tail(7).index, inplace=True)


# Save dataframe to file data.csv
data.to_csv(r'Data\data.csv')
data1.to_csv(r'Data\data1.csv')
data2.to_csv(r'Data\data2.csv')
data3.to_csv(r'Data\data3.csv')
data4.to_csv(r'Data\data4.csv')
data5.to_csv(r'Data\data5.csv')
data6.to_csv(r'Data\data6.csv')
data7.to_csv(r'Data\data7.csv')