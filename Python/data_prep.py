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
"""
rel_vars = ['ICU_Inflow', 'ICU_Inflow_SMA3d', 'ICU_Inflow_SMA7d',
            'Hosp_Inflow', 'Hosp_Inflow_SMA3d', 'Hosp_Inflow_SMA7d',
            'Tested', 'Tested_SMA3d', 'Tested_SMA7d',
            'Cases', 'Cases_SMA3d', 'Cases_SMA7d',
            'Cases_Pct', 'Cases_Pct_SMA3d', 'Cases_Pct_SMA7d',
            'RNA', 'RNA_SMA3d', 'RNA_SMA7d',
            'Vacc_Est', 'Vacc_Est_SMA3d', 'Vacc_Est_SMA7d',
            'Monday', 'Tuesday', 'Wednesday',
            'Thursday', 'Friday', 'Saturday']
"""
rel_vars = ['ICU_Inflow',
            'Hosp_Inflow',
            'Tested', 'Cases', 'Cases_Pct',
            'RNA',
            'Vacc_Est',
            'Monday', 'Tuesday', 'Wednesday',
            'Thursday', 'Friday', 'Saturday']
data = master.copy()
data = data[rel_vars]


# Take head of dataframe to Latex before making changes
vars_for_table = ['ICU_Inflow', 'Hosp_Inflow',
            'Tested', 'Cases', 'Cases_Pct',
            'RNA', 'Vacc_Est']
data_for_latex = data.loc['2021-01-05':'2021-01-09']
print(data_for_latex.to_latex(columns=vars_for_table, index=True, label="data_head",
                    caption="Sample of Data", position="h!"))


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
temp = data.loc[:, data.columns.difference(vars_excl)]
data.loc[:, data.columns.difference(vars_excl)] = temp.shift(1)

# Add additional features by lagging all regressors multiple time periods
for i in range(2, 15):
    temp_i = temp.shift(i)
    temp_i.columns = [f'{x}_Lag{i}' for x in temp_i.columns]
    data = data.join(temp_i)

# Insert lagged series of ICU Inflow
for i in range(1,15):
    data.insert(i, "ICU_Inflow_Lag{0}".format(i), data.ICU_Inflow.shift(i))

# Delete oldest x days due to nan's
data.drop(data.head(14).index, inplace=True)

# Save dataframe to file data.csv
data.to_csv(r'Data\data.csv')
